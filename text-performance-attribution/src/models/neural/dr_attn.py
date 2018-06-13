"""Deep Residualization model with attention.

This model uses an attentional RNN to encode the source into a vector e.

The attentional scores come from a separate network that looks at each
hidden state and decides the weight for that time step.

It also concatenates the confounds into a single vector, and passes
this through a few fully connected layers to predict the outcome.

Last, the model uses e to predict the residuals (errors) of the
confound=>outcome part of the model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import os

import tensorflow as tf

import sys; sys.path.append('../..')
import src.models.neural.inference_clients as inference_clients
import src.models.neural.tf_utils as tf_utils
import src.msc.utils as utils

# Disable TensorFlow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DrAttn(inference_clients.InferenceClient):
  """Deep Residualization model with attention."""

  @staticmethod
  def build_model_graph(config, params, dataset):
    graph = tf.Graph()
    with graph.as_default():
      iterators = dataset.make_tf_iterators(params['batch_size'])
      model = DrAttn(config, params, dataset, iterators)

    return tf_utils.TFModel(graph=graph, model=model, iterator=iterators)

  def __init__(self, config, params, dataset, iterators):
    """Constructs the graph and training/summary ops."""
    self.iter = iterators
    self.config = config
    self.params = params
    self.dataset = dataset

    self.learning_rate = tf.constant(params['learning_rate'])
    self.global_step = tf.Variable(0, trainable=False)
    self.dropout = tf.placeholder(tf.float32, name='dropout')

    source_name = dataset.input_varname()
    self.input_text, self.input_ids, self.input_lens = self.iter[source_name]

    # Use attention to encode the input.
    self.attn_scores, attn_context = self.attentional_encoder()
    # Now get all the confounds into one vector.
    confound_vector = self.vectorize_confounds()

    # Use confounds to predict targets.
    self.confound_output = defaultdict(dict)
    self.final_output = defaultdict(dict)
    for var in self.config.data_spec[1:]:
      if var['skip'] or var['control']:
        continue
      with tf.variable_scope(var['name']):
        if var['type'] == utils.CONTINUOUS:
          (confound_preds, confound_loss, final_preds,
           final_loss) = self.double_predict_regression(
               response=var,
               confound_input=confound_vector,
               x_input=attn_context)
        else:
          (confound_preds, confound_loss, final_preds,
           final_loss) = self.double_predict_classification(
               response=var,
               confound_input=confound_vector,
               x_input=attn_context)

      tf.summary.scalar('%s_confound_loss' % var['name'], confound_loss)
      self.confound_output[var['name']]['pred'] = confound_preds
      self.confound_output[var['name']]['loss'] = confound_loss

      tf.summary.scalar('%s_final_loss' % var['name'], final_loss)
      self.final_output[var['name']]['input'] = self.iter[var['name']]
      self.final_output[var['name']]['pred'] = final_preds
      self.final_output[var['name']]['loss'] = final_loss

    # Create loss ops.
    self.cum_confound_loss = tf.reduce_sum(
        [x['loss'] for x in self.confound_output.values()])
    self.cum_final_loss = tf.reduce_sum(
        [x['loss'] for x in self.final_output.values()])
    self.cumulative_loss = tf.reduce_sum(
        [self.cum_confound_loss, self.cum_final_loss])
    tf.summary.scalar('cum_confound_loss', self.cum_confound_loss)
    tf.summary.scalar('cum_final_loss', self.cum_final_loss)
    tf.summary.scalar('cum_loss', self.cumulative_loss)

    # Training ops.
    self.train_step = tf.contrib.layers.optimize_loss(
        loss=self.cumulative_loss,
        global_step=self.global_step,
        learning_rate=self.learning_rate,
        optimizer='Adam',
        summaries=['loss', 'gradient_norm'])

    self.summaries = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())
    self.trainable_variable_names = [v.name for v in tf.trainable_variables()]

  def attentional_encoder(self):
    """Builds an encoder which uses attention over the source.

    TODO(rpryzant) -- conbine with a_attn.py::attentional_encoder.

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
    with tf.variable_scope('attention'):
      attn_scores, attn_context = tf_utils.attention(
          states=rnn_outputs,
          seq_lens=self.input_lens,
          layers=self.params['attn_layers'],
          units=self.params['attn_units'],
          dropout=self.dropout)

    return attn_scores, attn_context

  def vectorize_confounds(self):
    """Concats all of the confounds into  a single vector.

    TODO: combine with dr_bow::vectorize_confounds

    For example, if there are two confounds, a scalar named C_a and a binary
    categorical variable named C_b, then the returned vector will be
      [C_a, C_b_indicator_0, C_b_indicator_1]

    Returns:
      confound_vector: tensor [batch size, num confounds * levels per confound]
    """
    confounds = []
    for var in self.config.data_spec[1:]:
      if var['skip'] or not var['control']:
        continue
      if var['type'] == utils.CONTINUOUS:
        vals_as_cols = tf.expand_dims(self.iter[var['name']], 1)
        confounds.append(vals_as_cols)

      else:
        confound_embeddings = tf.get_variable(
            name='%s_embeddings' % var['name'],
            shape=[
                self.dataset.num_levels(var['name']),
                self.params['embedding_size']
            ])
        confounds.append(
            tf.nn.embedding_lookup(confound_embeddings, self.iter[var['name']]))
    confound_vector = tf.concat(confounds, axis=1)
    return confound_vector

  def double_predict_regression(self, response, confound_input, x_input):
    """Predicts a scalar outcome twice.

    First, predict the outcome from the confound.
    Second, concat the text input and the residuals from step (1), and
    use the resulting vector to generate better predictions of the outcome.

    Args:
      response: dict, an element of config.variable_spec. This is the
        response variable we are predicting.
      confound_input: tensor [batch, num confounds], there are the
        confounds, all stacked up into vectors.
      x_input: tensor [batch, hidden], an encoded version of the text input.

    Returns:
      confound_preds: tensor [batch], the predictions from the confound.
      confound_loss: tensor [batch], the l2 loss between confound
        predictions and targets.
      final_preds: tensor[batch], the final predictions from confound
        residuals + text input.
      final_loss: tensor[batch], the l2 loss between final preds and targets.
    """
    with tf.variable_scope('control_pred'):
      confound_preds, confound_loss = tf_utils.regressor(
          inputs=confound_input,
          labels=self.iter[response['name']],
          layers=self.params['regressor_layers'],
          hidden=self.params['regressor_units'],
          dropout=self.dropout)

    # force this into [batch size, embedding width + 1]
    final_input = tf.concat(
        [tf.expand_dims(confound_preds, 1), x_input], axis=1)

    if self.params['encoder_layers'] == 0:
      if self.params['use_glove']:
        # TODO(rpryzant) -- dynamically set this value. Don't hardcode it.
        x_input_dim = 50
      else:
        x_input_dim = self.params['embedding_size']
    else:
      x_input_dim = self.params['encoder_units'] * 2

    final_input = tf.reshape(final_input, [-1, x_input_dim + 1])

    with tf.variable_scope('final_pred'):
      final_preds, final_loss = tf_utils.regressor(
          inputs=final_input,
          labels=self.iter[response['name']],
          layers=self.params['regressor_layers'],
          hidden=self.params['regressor_units'],
          dropout=self.dropout)

    return confound_preds, confound_loss, final_preds, final_loss

  def double_predict_classification(self, response, confound_input, x_input):
    """Predicts a categorical outcome twice.

    First, predict the outcome from the confound.
    Second, concat the text input and the residuals from step (1), and
    use the resulting vector to generate better predictions of the outcome.

    Args:
      response: dict, an element of config.variable_spec. This is the
        response variable we are predicting.
      confound_input: tensor [batch, num confounds], this is the
        confounds, all stacked up into vectors.
      x_input: tensor [batch, hidden], an encoded version of the text input.
    Returns:
      confound_preds: tensor [batch, classes], predictions from the confound.
      confound_loss: tensor [batch], cross-entropy loss between confound
        predictions and targets.
      final_preds: tensor[batch, classes], final predictions from confound
        residuals + text input.
      final_loss: tensor[batch, classes], cross-entropy loss between final
        preds and targets.
    """
    with tf.variable_scope('control_pred'):
      confound_preds, confound_loss = tf_utils.classifier(
          inputs=confound_input,
          labels=self.iter[response['name']],
          layers=self.params['classifier_layers'],
          num_classes=self.dataset.num_levels(response['name']),
          hidden=self.params['classifier_units'],
          dropout=self.dropout,
          sparse_labels=True)

    final_input = tf.concat([confound_preds, x_input], axis=1)

    with tf.variable_scope('final_pred'):
      final_preds, final_loss = tf_utils.classifier(
          inputs=final_input,
          labels=self.iter[response['name']],
          layers=self.params['classifier_layers'],
          num_classes=self.dataset.num_levels(response['name']),
          hidden=self.params['classifier_units'],
          dropout=self.dropout,
          sparse_labels=True)

    return confound_preds, confound_loss, final_preds, final_loss

  def train(self, sess):
    """Trains the model for a batch."""
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
    return self.attn_model_inference(sess, self.input_text, self.final_output,
                                     self.attn_scores)
