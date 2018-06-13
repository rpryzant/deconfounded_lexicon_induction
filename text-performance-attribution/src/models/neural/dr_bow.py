"""Deep Residualization with bag of words text encodings.

This model uses a bag-of-words representation followed by several feed-forward
layers to encode the source into a vector e.

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


class DrBow(inference_clients.InferenceClient):
  """Deep Residualization with bag of words text encodings."""

  @staticmethod
  def build_model_graph(config, params, dataset):
    graph = tf.Graph()
    with graph.as_default():
      iterators = dataset.make_tf_iterators(params['batch_size'])
      model = DrBow(config, params, dataset, iterators)

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

    # Transform input text into a big BOW vector.
    with tf.variable_scope('input'):
      input_vector = tf.map_fn(
          lambda seq: tf_utils.sparse_to_dense_vector(  # pylint: disable=g-long-lambda
              seq, self.dataset.vocab_size),
          self.iter[dataset.input_varname()][1])
      input_encoded = tf_utils.fc_tube(
          inputs=tf.cast(input_vector, tf.float32),
          num_outputs=self.params['encoding_dim'],
          layers=self.params['encoder_layers'],
          bias=False,
          use_relu=False)

    # Pull out the vector of weights which dots the input vector.
    cur_graph = tf.get_default_graph()
    self.feature_weights = cur_graph.get_tensor_by_name(
        'input/layer_0/weights:0')

    # Transform the confounds similarly.
    with tf.variable_scope('confound_input'):
      confound_vector = self.vectorize_confounds()

    # Use confounds to predict the targets.
    self.confound_output = defaultdict(dict)
    self.final_output = defaultdict(dict)
    for var in self.config.data_spec[1:]:
      if var['skip'] or var['control']:
        continue
      with tf.variable_scope(var['name']):
        if var['type'] == utils.CONTINUOUS:
          confound_preds, confound_loss, final_preds, final_loss = \
              self.double_predict_regression(
                  response=var,
                  confound_input=confound_vector,
                  x_input=input_encoded)
        else:
          confound_preds, confound_loss, final_preds, final_loss = \
              self.double_predict_classification(
                  response=var,
                  confound_input=confound_vector,
                  x_input=input_encoded)

      tf.summary.scalar('%s_confound_loss' % var['name'], confound_loss)
      self.confound_output[var['name']]['pred'] = confound_preds
      self.confound_output[var['name']]['loss'] = confound_loss

      tf.summary.scalar('%s_final_loss' % var['name'], final_loss)
      self.final_output[var['name']]['input'] = self.iter[var['name']]
      self.final_output[var['name']]['pred'] = final_preds
      self.final_output[var['name']]['loss'] = final_loss
      self.final_output[var['name']]['weights'] = cur_graph.get_tensor_by_name(
          '%s/final_pred/prediction_head/weights:0' % var['name'])

    # Regularization.
    if self.params['lambda'] > 0:
      if self.params['regularizer'] == 'l2':
        regularizer = tf.contrib.layers.l2_regularizer(self.params['lambda'])
      else:
        regularizer = tf.contrib.layers.l1_regularizer(self.params['lambda'])

      if self.params['reg_type'] == 'all':
        regularized_weights = tf.trainable_variables()
      else:
        regularized_weights = [self.feature_weights]

      regularization_term = tf.contrib.layers.apply_regularization(
          regularizer, regularized_weights)
    else:
      regularization_term = 0

    # Loss ops.
    self.cum_confound_loss = tf.reduce_sum(
        [x['loss'] for x in self.confound_output.values()])
    self.cum_final_loss = tf.reduce_sum(
        [x['loss'] for x in self.final_output.values()])
    self.cumulative_loss = tf.reduce_sum(
        [self.cum_confound_loss, self.cum_final_loss])
    self.cumulative_loss += regularization_term

    tf.summary.scalar('regularization_loss', regularization_term)
    tf.summary.scalar('cum_confound_loss', self.cum_confound_loss)
    tf.summary.scalar('cum_final_loss', self.cum_final_loss)
    tf.summary.scalar('cum_loss', self.cumulative_loss)

    # Training ops.
    self.train_step = tf.contrib.layers.optimize_loss(
        loss=self.cumulative_loss,
        global_step=self.global_step,
        learning_rate=self.learning_rate,
        optimizer='SGD',
        summaries=['loss', 'gradient_norm'])

    self.summaries = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())
    self.trainable_variable_names = [v.name for v in tf.trainable_variables()]

  def double_predict_regression(self, response, confound_input, x_input):
    """Predicts a scalar outcome twice.

    First, predict the outcome from the confound.
    Second, concat the text input and the residuals from step (1), and
    use the resulting vector to generate better predictions of the outcome.

    Args:
      response: dict, an element of config.variable_spec. This is the
        response variable we are predicting.
      confound_input: tensor [batch, num confounds], the confounds, all
        stacked up into vectors.
      x_input: tensor [batch, hidden], an encoded version of the text input.

    Returns:
      confound_preds: tensor [batch], the predictions from the confound.
      confound_loss: tensor [batch], the l2 loss between confound
        predictions and targets.
      final_preds: tensor[batch], final predictions from confound
        residuals + text input.
      final_loss: tensor[batch], the l2 loss between final preds and targets.
    """
    with tf.variable_scope('control_pred'):
      confound_preds, confound_loss = tf_utils.regressor(
          inputs=confound_input,
          labels=self.iter[response['name']],
          layers=self.params['regression_layers_1'],
          hidden=self.params['regression_hidden_1'],
          dropout=self.dropout)
      confound_preds = tf.expand_dims(confound_preds, 1)

    # force this into [batch size, attn width + 1]
    final_input = tf.concat([x_input, confound_preds], axis=1)
    final_input = tf.reshape(final_input, [-1, self.params['encoding_dim'] + 1])

    with tf.variable_scope('final_pred'):
      final_preds, final_loss = tf_utils.regressor(
          bias=False,
          inputs=final_input,
          labels=self.iter[response['name']],
          layers=self.params['regression_layers_2'],
          hidden=self.params['regression_hidden_2'],
          dropout=self.dropout)

    return confound_preds, confound_loss, final_preds, final_loss

  def double_predict_classification(self, response, confound_input, x_input):
    """Predicts a categorical outcome twice.

    First, predict the outcome from the confound.
    Second, concat the text input and the residuals from step (1), and
    use the resulting vector to generate better predictions of the outcome.

    Args:
      response: dict, an element of config.variable_spec. This is the response
        variable we are predicting.
      confound_input: tensor [batch, num confounds], the confounds, all
        stacked up into vectors.
      x_input: tensor [batch, hidden], an encoded version of the text input.

    Returns:
      confound_preds: tensor [batch, num classes], the predictions from the
        confound.
      confound_loss: tensor [batch], the cross-entropy loss between confound
        predictions and targets.
      final_preds: tensor[batch, num classes], the final predictions from
        confound residuals + text input.
      final_loss: tensor[batch, num classes], the cross-entropy loss between the
        final preds and targets.
    """
    with tf.variable_scope('control_pred'):
      confound_preds, confound_loss = tf_utils.classifier(
          inputs=confound_input,
          labels=self.iter[response['name']],
          layers=self.params['classification_layers_1'],
          num_classes=self.dataset.num_levels(response['name']),
          hidden=self.params['classification_hidden_1'],
          dropout=self.dropout,
          sparse_labels=True)

    if self.params['ablate_confounds']:
      confound_preds = tf.zeros_like(confound_preds)

    final_input = tf.concat([x_input, confound_preds], axis=1)

    with tf.variable_scope('final_pred'):
      final_preds, final_loss = tf_utils.classifier(
          bias=False,
          inputs=final_input,
          labels=self.iter[response['name']],
          layers=self.params['classification_layers_2'],
          num_classes=self.dataset.num_levels(response['name']),
          hidden=self.params['classification_hidden_2'],
          dropout=self.dropout,
          sparse_labels=True)

    return confound_preds, confound_loss, final_preds, final_loss

  def vectorize_confounds(self):
    """Concats all of the confounds into  a single vector.

    TODO -- combine with dr_attn::vectorize_confounds()

    For example, if there are two confounds, a scalar named C_a and a binary
    categorical variable named C_b, then the returned vector will be
      [C_a, C_b_indicator_0, C_b_indicator_1]

    Returns:
      confound_vector: tensor [batch size, num confounds * levels per confound]
    """
    confounds = []
    for variable in self.config.data_spec[1:]:
      if variable['skip'] or not variable['control']:
        continue
      if variable['type'] == utils.CONTINUOUS:
        confounds.append(tf.expand_dims(self.iter[variable['name']], 1))
      else:

        col_per_example = tf.expand_dims(self.iter[variable['name']], 1)
        one_hot_confound_vectors = tf.map_fn(
            lambda level: tf_utils.sparse_to_dense_vector(  # pylint: disable=g-long-lambda
                level, self.dataset.num_levels(variable['name'])),  # pylint: disable=cell-var-from-loop
            col_per_example)
        confounds.append(tf.cast(one_hot_confound_vectors, tf.float32))

    confound_vecs = tf.concat(confounds, axis=1)
    return confound_vecs

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
                                                    on true positives ONLY.
    """
    return self.bow_model_inference(sess, self.feature_weights,
                                    self.final_output)
