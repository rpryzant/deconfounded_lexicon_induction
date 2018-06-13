"""Adversarial Attention model.

This model uses an attentional RNN to encode the source into a vector e.

The attentional scores come from a separate network that looks at each
hidden state and decides the weight for that time step.

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


class AAttn(inference_clients.InferenceClient):
  """Adversarial Attention model."""

  @staticmethod
  def build_model_graph(config, params, dataset):
    graph = tf.Graph()
    with graph.as_default():
      iterators = dataset.make_tf_iterators(params['batch_size'])
      model = AAttn(config, params, dataset, iterators)

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
    self.input_text, self.input_ids, self.input_lens = self.iter[source_name]

    # Use attention to encode the source.
    self.attn_scores, attn_context = self.attentional_encoder()

    # Now build all the prediction heads (one per non-input variable).
    self.step_output = defaultdict(dict)
    for variable in self.config.data_spec[1:]:
      if variable['skip']:
        continue

      with tf.variable_scope(variable['name'] + '_prediction_head'):
        if variable['control']:
          prediction_input = self.reverse(attn_context)
        else:
          prediction_input = tf.identity(attn_context)

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

  def attentional_encoder(self):
    """Builds an encoder which uses attention over the source.

    TODO(rpryzant) -- conbine with dr_attn.py::attentional_encoder.

    Returns:
      attn_scores: tensor [batch size, max seq len], probability
        distributions over the tokens of each input sequence.
      attn_context: tensor [batch size, hidden size], weighted average of the
        hidden states as decided by attn_scores.
    """
    with tf.variable_scope('encoder'):
      rnn_outputs, _ = tf_utils.rnn_encode(
          source=self.input_ids,
          source_len=self.input_lens,
          vocab_size=self.dataset.vocab_size,
          embedding_size=self.params['embedding_size'],
          layers=self.params['encoder_layers'],
          units=self.params['encoder_units'],
          dropout=self.dropout)

    # Attention over the source.
    with tf.variable_scope('attention'):
      attn_scores, attn_context = tf_utils.attention(
          states=rnn_outputs,
          seq_lens=self.input_lens,
          layers=self.params['attn_layers'],
          units=self.params['attn_units'],
          dropout=self.dropout)
    return attn_scores, attn_context

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
    return self.attn_model_inference(sess, self.input_text, self.step_output,
                                     self.attn_scores)
