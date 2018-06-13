"""Adversarial bag of words model.

This model uses a bag-of-words representation followed by several feed-forward
layers to encode the source into a vector e.

e is then passed to (1) an adversarial discriminator for each confounder, and
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


class ABow(inference_clients.InferenceClient):
  """Adversarial bag of words model."""

  @staticmethod
  def build_model_graph(config, params, dataset):
    graph = tf.Graph()
    with graph.as_default():
      iterators = dataset.make_tf_iterators(params['batch_size'])
      model = ABow(config, params, dataset, iterators)

    return tf_utils.TFModel(graph=graph, model=model, iterator=iterators)

  def __init__(self, config, params, dataset, iterators):
    """Constructs the graph and training/summary ops."""
    self.iter = iterators
    self.config = config
    self.params = params
    self.dataset = dataset

    self.learning_rate = tf.constant(params['learning_rate'])
    self.dropout = tf.placeholder(tf.float32, name='dropout')
    self.global_step = tf.Variable(0, trainable=False)

    source_name = dataset.input_varname()
    self.input_text, _, _ = self.iter[source_name]

    # Transform the input text into a big bag of words vector.
    with tf.variable_scope('input'):
      input_vector = tf.map_fn(
          lambda seq: tf_utils.sparse_to_dense_vector(  # pylint: disable=g-long-lambda
              seq, self.dataset.vocab_size),
          self.iter[dataset.input_varname()][1])
      input_encoded = tf_utils.fc_tube(
          inputs=tf.cast(input_vector, tf.float32),
          num_outputs=self.params['encoder_layers'],
          layers=self.params['encoder_layers'])

    # Pull out the vector of weights which dots the input vector.
    # TODO(rpryzant) -- there must be a more elegant way to do this in TF?
    cur_graph = tf.get_default_graph()
    self.feature_weights = cur_graph.get_tensor_by_name(
        'input/layer_0/weights:0')
    self.feature_intercept = cur_graph.get_tensor_by_name(
        'input/layer_0/biases:0')

    # Now build all the prediction heads, one for each non-input variable.
    self.step_output = defaultdict(dict)
    for variable in self.config.data_spec[1:]:
      if variable['skip']:
        continue

      with tf.variable_scope(variable['name'] + '_prediction_head'):
        if variable['control']:
          prediction_input = self.reverse(input_encoded)
        else:
          prediction_input = tf.identity(input_encoded)

        if variable['type'] == utils.CATEGORICAL:
          preds, mean_loss = tf_utils.classifier(
              inputs=prediction_input,
              labels=self.iter[variable['name']],
              layers=self.params['classifier_layers'],
              num_classes=self.dataset.num_levels(variable['name']),
              hidden=self.params['classifier_units'],
              dropout=self.dropout,
              sparse_labels=True)
        elif variable['type'] == utils.CONTINUOUS:
          preds, mean_loss = tf_utils.regressor(
              inputs=prediction_input,
              labels=self.iter[variable['name']],
              layers=self.params['regressor_layers'],
              hidden=self.params['regressor_units'],
              dropout=self.dropout)
        else:
          raise Exception('ERROR: unknown type %s for variable %s' %
                          (variable['type'], variable['name']))

        mean_loss = tf.scalar_mul(variable['weight'], mean_loss)

      tf.summary.scalar('%s_loss' % variable['name'], mean_loss)
      self.step_output[variable['name']]['input'] = self.iter[variable['name']]
      self.step_output[variable['name']]['loss'] = mean_loss
      self.step_output[variable['name']]['pred'] = preds

    # Regularize the parameters.
    if self.params['lambda'] > 0:
      if self.params['regularizer'] == 'l2':
        regularizer = tf.contrib.layers.l2_regularizer(self.params['lambda'])
      else:
        regularizer = tf.contrib.layers.l1_regularizer(self.params['lambda'])

      if self.params['reg_type'] == 'all':
        regularization_weights = tf.trainable_variables()
      else:
        regularization_weights = [self.feature_weights]

      regularization_term = tf.contrib.layers.apply_regularization(
          regularizer, regularization_weights)
    else:
      regularization_term = 0

    tf.summary.scalar('regularization_loss', regularization_term)

    # Optimization ops.
    self.loss = tf.reduce_sum([x['loss'] for x in self.step_output.values()])
    self.loss += regularization_term
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

  def reverse(self, in_tensor):
    """Reverses the gradients of a tensor of any shape."""
    input_shape = in_tensor.get_shape()
    out_tensor = reverse_grad(in_tensor)
    out_tensor.set_shape(input_shape)
    return out_tensor

  def train(self, sess):
    """Trains for a batch."""
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
                                                    on true positives ONLY.
    """
    return self.bow_model_inference(sess, self.feature_weights,
                                    self.step_output)
