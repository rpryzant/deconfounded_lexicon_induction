"""Adversarial CNN model.

This model uses an RNN and CNN to encode the source into a vector `e`.
Note that using the RNN is optional, and you can skip it (and have the CNN
look directly at the word vectors) by setting the `encoder_layers` parameter
to 0 in your config.yaml. This setting (without the RNN) appears to perform
better.

The model decides ngram "importance" by measuring the size of the activations
coming from each filter.

`e` is then passed to (1) an adversarial discriminator for each confounder, and
(2) a prediction head for each outcome.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import os

import tensorflow as tf

from tensorflow.python.framework import function

import sys; sys.path.append('../..')
import src.models.neural.inference_clients as inference_clients
import src.models.neural.tf_utils as tf_utils
import src.msc.utils as utils

# Disable TensorFlow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# # # # global gradient reversal functions  # # # #
def reverse_grad_grad(_, grad):
  return tf.constant(-1.) * grad


@function.Defun(tf.float32, python_grad_func=reverse_grad_grad)
def reverse_grad(tensor):
  return tf.identity(tensor)


# # # # # # # # # # # # # # # # # # # # # # # # # #


class ACnn(inference_clients.InferenceClient):
  """Adversarial CNN model."""

  @staticmethod
  def build_model_graph(config, params, dataset):
    graph = tf.Graph()
    with graph.as_default():
      iterators = dataset.make_tf_iterators(
          params['batch_size'], undo_ngrams=True)
      model = ACnn(config, params, dataset, iterators)

    return tf_utils.TFModel(graph=graph, model=model, iterator=iterators)

  def __init__(self, config, params, dataset, iterators):
    """Constructs the graph and training/summary ops."""
    self.iter = iterators
    self.config = config
    self.params = params
    self.dataset = dataset
    self.filter_sizes = [int(x) for x in self.params['filter_size'].split(',')]

    tf_graph = tf.get_default_graph()

    self.learning_rate = tf.constant(params['learning_rate'])
    self.dropout = tf.placeholder(tf.float32, name='dropout')
    self.global_step = tf.Variable(0, trainable=False)

    source_name = dataset.input_varname()
    self.input_text, self.input_ids, self.input_lens = self.iter[source_name]

    # Use a cnn to encode the source.
    conv, src_encoded = self.cnn_encoder()

    # Now build all the prediction heads (one per non-input variable).
    self.step_output = defaultdict(dict)
    for variable in self.config.data_spec[1:]:
      if variable['skip']:
        continue

      with tf.variable_scope(variable['name']):
        if variable['control']:
          prediction_input = self.reverse(src_encoded)
        else:
          prediction_input = tf.identity(src_encoded)

        # Each prediction head is a single fully-connected layer without
        # activation functions or bias. This makes it a simple linear projection
        # into the output space.
        if variable['type'] == utils.CATEGORICAL:
          preds, mean_loss = tf_utils.classifier(
              inputs=prediction_input,
              labels=self.iter[variable['name']],
              layers=1,
              num_classes=self.dataset.num_levels(variable['name']),
              dropout=self.dropout,
              sparse_labels=True,
              bias=False)
        elif variable['type'] == utils.CONTINUOUS:
          preds, mean_loss = tf_utils.regressor(
              inputs=prediction_input,
              labels=self.iter[variable['name']],
              layers=1,
              dropout=self.dropout,
              bias=False)
        else:
          raise Exception('ERROR: unknown type %s for variable %s' %
                          (variable['type'], variable['name']))

        prediction_head_weights = tf_graph.get_tensor_by_name(
            '%s/prediction_head/weights:0' % variable['name'])

        mean_loss = variable['weight'] * mean_loss
        # The user is allowed to specify a "rho" term which is a dampening
        # factor on the adversarial signal. This helps the model achieve a
        # balance between the losses of the prediction head and encoder.
        if variable['control']:
          mean_loss = self.params['rho'] * mean_loss

      tf.summary.scalar('%s_loss' % variable['name'], mean_loss)

      # Save everything you need for inference: the input, loss, the
      # convolutional feature maps, the output projection weights, and the
      # model's predictions.
      self.step_output[variable['name']]['input'] = self.iter[variable['name']]
      self.step_output[variable['name']]['loss'] = mean_loss
      self.step_output[variable['name']]['conv'] = conv
      self.step_output[variable['name']]['weights'] = prediction_head_weights
      self.step_output[variable['name']]['pred'] = preds

    # Optimization and summary writing.
    self.loss = tf.reduce_sum([x['loss'] for x in self.step_output.values()])
    tf.summary.scalar('global_loss', self.loss)

    self.train_step = tf.contrib.layers.optimize_loss(
        loss=self.loss,
        global_step=self.global_step,
        learning_rate=self.learning_rate,
        clip_gradients=self.params['gradient_clip'],
        optimizer='Adam',
        summaries=['gradient_norm'])

    # Savers, summaries, etc.
    self.trainable_variable_names = [v.name for v in tf.trainable_variables()]
    self.summaries = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())

  def cnn_encoder(self):
    """Builds an encoder which runs convolutions over an rnn.

    Note that using the RNN is optional, and you can skip it (and have the CNN
    look directly at the word vectors) by setting the `encoder_layers` parameter
    to 0 in your config.yaml. This setting (without the RNN) appears to perform
    better.


    Returns:
      convs: list(tensor [batch, time - (filter size - 1), n filters]),
        convolutional feature maps for each filter size.
      src_encoded: tensor [batch size, num filters * num filter sizes], the
        feature maps after average pooling across the time dimension.
    """
    # Use an RNN to enrich the word vector inputs.
    with tf.variable_scope('encoder'):
      rnn_outputs, _ = tf_utils.rnn_encode(
          source=self.input_ids,
          source_len=self.input_lens,
          vocab_size=self.dataset.vocab_size,
          embedding_size=self.params['embedding_size'],
          layers=self.params['encoder_layers'],
          units=self.params['encoder_units'],
          dropout=self.dropout)
    # Add a 4th dimension to rnn_inputs in preparation for 2d convolution
    # (now the convolution inputs have 1 channel.
    rnn_outputs = tf.expand_dims(rnn_outputs, -1)

    # Run params['n_filters'] filters of size self.filter_sizes horizontally
    # over the encoded source.
    # The 2nd dimension is dynamically set because the "height" of the
    # convolution inputs is encoder_units *2 if we use a bi-lstm and
    # embedding_size if we encoder_layers=0 (i.e., if we skipped the lstm).
    # The 3rd dimension has size 1 because the size of our filter matches
    # that of the input along that axis. In other words, we're sliding
    # our filters horizontally across each sequence.
    rnn_output_dim = rnn_outputs.get_shape()[-2]
    outputs = []
    convs = []
    for size in self.filter_sizes:
      with tf.variable_scope('conv_avgpool_%s' % size):
        w_conv = tf.get_variable(
            'w_conv', [size, rnn_output_dim, 1, self.params['n_filters']])
        conv = tf.nn.conv2d(
            rnn_outputs, w_conv, strides=[1, 1, 1, 1], padding='VALID')
        # Get rid of the vertical axis, now conv is [batch, time, filters].
        conv = tf.squeeze(conv)
        conv = tf.nn.relu(conv)
        conv = tf.nn.dropout(conv, 1 - self.dropout)
        # Global average pooling.
        pooled = tf.reduce_mean(conv, axis=1)
        pooled = tf.reshape(pooled, [-1, self.params['n_filters']])

        convs.append(conv)
        outputs.append(pooled)

    outputs = tf.concat(outputs, 1)

    return convs, outputs

  def reverse(self, in_tensor):
    """Reverses the gradients in a tensor of any shape."""
    input_shape = in_tensor.get_shape()
    out_tensor = reverse_grad(in_tensor)
    out_tensor.set_shape(input_shape)
    return out_tensor

  def train(self, sess):
    """Trains on a batch."""
    ops = [self.global_step, self.train_step, self.summaries]
    return sess.run(ops, feed_dict={self.dropout: self.params['dropout']})

  def inference_on_batch(self, sess):
    """Performs inference on a batch of inputs.

    Args:
      sess: tf.Session, the current TensorFlow session.

    Returns:
      predictions: dict(string => list(float) or list(list(float)). A mapping
        from variable to predictions or logits for each example in the batch.
      token_importance: dict(string => dict(string => list(float))) or
        dict(string => dict(string => dict(string => list(float)))).
        For continuous variables:
          variable name => feature name => list of attention scores.
        For categorical variables:
          variable name => level => feature name => list of attention scores
    """
    return self.cnn_model_inference(sess, self.input_text, self.step_output)
