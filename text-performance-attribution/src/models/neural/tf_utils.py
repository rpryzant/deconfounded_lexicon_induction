"""TensorFlow utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import os
import tensorflow as tf

# Disable TensorFlow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# namedtuple for keeping models, graphs, and iterators as one unit.
TFModel = namedtuple('Model', ('graph', 'model', 'iterator'))


def fc_tube(inputs, num_outputs, layers, dropout=0.0, bias=True, use_relu=True):
  """Builds one or more fully connected layers.

  TODO -- replace with layers_core.Dense, which has become part of TF
  since this was written.
  TODO -- try batch norm, see if it helps.

  Args:
    inputs: tensor, the inputs to this neural network.
    num_outputs: int, the number of outputs of this neural network. This is also
      used as the hidden size.
    layers: int, number of fully connected layers to use.
    dropout: float, dropout rate.
    bias: bool, whether to use a bias or not.
    use_relu: bool, whether to use a relu or not.

  Returns:
    x: tensor [batch size, num_outputs], the tranformed input after being passed
      through one or more fully connected layers.
  """
  if layers == 0:
    return inputs

  x = tf.contrib.layers.fully_connected(
      inputs=inputs,
      num_outputs=num_outputs,
      biases_initializer=tf.zeros_initializer() if bias else None,
      activation_fn=tf.nn.relu,
      scope='layer_0')
  x = tf.nn.dropout(x, (1 - dropout))
  if use_relu:
    x = tf.nn.relu(x)

  for layer in range(layers - 1):
    x = tf.contrib.layers.fully_connected(
        inputs=x,
        biases_initializer=tf.zeros_initializer() if bias else None,
        num_outputs=num_outputs,
        activation_fn=tf.nn.relu,
        scope='layer_%d' % (layer + 1))
    x = tf.nn.dropout(x, (1 - dropout))
    if use_relu:
      x = tf.nn.relu(x)

  return x


def regressor(inputs, labels, layers, hidden=128, dropout=0.0, bias=True):
  """Builds a neural network for predicting scalar targets.

  Args:
    inputs: tensor [batch size, num features], the inputs to this network.
    labels: tensor [batch size], the regression targets.
    layers: int, the number of layers.
    hidden: int, the number of hidden units
    dropout: float, the dropout rate.
    bias: bool, whether to use a bias.

  Returns:
    preds: [batch size], predictions for the targets.
    mean_loss: tensor [int], average l2 loss for the predictions.
  """
  x = fc_tube(inputs, hidden, layers - 1, dropout)
  preds = tf.contrib.layers.fully_connected(
      inputs=x,
      num_outputs=1,
      activation_fn=None,
      biases_initializer=tf.zeros_initializer() if bias else None,
      scope='prediction_head')
  preds = tf.squeeze(preds)
  losses = tf.nn.l2_loss(preds - labels)
  mean_loss = tf.reduce_mean(losses)

  return preds, mean_loss


def classifier(inputs,
               labels,
               layers,
               num_classes,
               hidden=128,
               dropout=0.0,
               sparse_labels=False,
               bias=True):
  """Builds a neural network for predicting categorical targets.

  Args:
    inputs: tensor [batch size, num features], the inputs to this network.
    labels: tensor [batch size], classification target ids.
    layers: int, the number of layers.
    num_classes: int, the number of classes for the variable that
      is being predicted.
    hidden: int, the number of hidden units to use.
    dropout: float, dropout rate.
    sparse_labels: bool, whether the labels are sparse (one-hot) or dense (IDs).
    bias: bool, whether to use a bias.

  Returns:
    logits: [batch size, num_classes], the predicted likelihood for each class.
    mean_loss: [float], the average cross entropy between the logits and labels.
  """
  x = fc_tube(inputs, hidden, layers - 1, dropout)
  logits = tf.contrib.layers.fully_connected(
      inputs=x,
      num_outputs=num_classes,
      biases_initializer=tf.zeros_initializer() if bias else None,
      activation_fn=None,
      scope='prediction_head')

  labels = tf.cast(labels, tf.int32)

  # Mean log perplexity per batch.
  if sparse_labels:
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
  else:
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

  mean_loss = tf.reduce_mean(losses)

  return logits, mean_loss


def build_rnn_cells(layers=1, units=256, dropout=0.0):
  """Build one or more RNN cells."""

  def single_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(units)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell=cell, input_keep_prob=(1.0 - dropout))
    return cell

  cells = [single_cell() for _ in range(layers)]
  multicell = tf.contrib.rnn.MultiRNNCell(cells)
  return multicell


def rnn_encode(source, source_len, vocab_size, embedding_size, layers, units,
               dropout):
  """Uses an RNN to encode some text.

  Args:
    source: tensor [batch size, max seq len], sequences of IDs corresponding to
      the padded text input.
    source_len: tensor [batch size], The length of each source sequence.
    vocab_size: int, the number of vocabulary tokens.
    embedding_size: int, the size of embeddings being fed into the rnn.
    layers: int: number of layers to use in the rnn.
    units: int, LSTM cell hidden size.
    dropout: int, dropout rate (applied to inputs at each timestep).

  Returns:
    hidden_states: tensor [batch, max seq len, hidden size], the hidden states
      of the RNN at each time step.
    source_embedded: tensor [batch, hidden_size], the final hidden state of
      the encoder.
  """
  with tf.variable_scope('embedding'):
    embeddings = tf.get_variable(name='E', shape=[vocab_size, embedding_size])
    source_embedded = tf.nn.embedding_lookup(embeddings, source)

  # Skip the RNN and simply use word vectors as our encoding of the source.
  if layers == 0:
    return source_embedded, source_embedded

  with tf.variable_scope('encoder'):
    cells_fw = build_rnn_cells(layers, units, dropout)
    cells_bw = build_rnn_cells(layers, units, dropout)
    bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        cells_fw,
        cells_bw,
        source_embedded,
        dtype=tf.float32,
        sequence_length=source_len)
    hidden_states = tf.concat(bi_outputs, -1)

  return hidden_states, source_embedded


def attention(states, seq_lens, layers=1, units=128, dropout=0.0):
  """Computes attention over a sequence of hidden states.

  We get scores from each hidden state by passing it through a feedforward
  whose weights are shared across timesteps.

  Args:
    states: tensor [batch size, max seq len, hidden state size], the sequence
      of hidden states that we want to aggregate over.
    seq_lens: tensor [batch size], the length of each sequence in `states`.
      This variable is used to zero out the attentional scores of pad tokens.
    layers: int, how many layers to use in the scoring network.
    units: int, hidden size of the scoring network.
    dropout: float, dropout rate.
  Returns:
    scores_normalized: tensor [batch size, max seq len], probability
      distributions over the tokens of each input sequence.
    context: tensor [batch size, hidden size], weighted average of `states`
      where the weights are decided by scores_normalized.
  """
  state_size = states.get_shape().as_list()[-1]

  x = fc_tube(
      inputs=states, num_outputs=units, layers=layers - 1, dropout=dropout)
  scores = tf.contrib.layers.fully_connected(
      inputs=x, num_outputs=1, activation_fn=None, scope='attn_score')
  scores = tf.squeeze(scores)

  # Replace all scores for padded inputs with tf.float32.min.
  scores_mask = tf.sequence_mask(
      lengths=tf.to_int32(seq_lens),
      maxlen=tf.to_int32(tf.reduce_max(seq_lens)),
      dtype=tf.float32)
  scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)

  # Normalize the scores.
  scores_normalized = tf.nn.softmax(scores, name='scores_normalized')

  # Calculate the weighted average of the attention inputs
  # according to the scores, then reshape to make one vector per example.
  context = tf.expand_dims(scores_normalized, 2) * states
  context = tf.reduce_sum(context, 1, name='context')
  context.set_shape([None, state_size])

  return scores_normalized, context


def sparse_to_dense_vector(sparse_indices, total_features):
  """Converts a tensor of token IDs to bag-of-words vectors.

  For example, if sparse_indices = [0, 3, 3, 1] and total_features = 4 then
  the returned tensor will be [1, 0, 1, 1].

  Args:
    sparse_indices: tensor [seq len], the sequence of token IDs we want
      to convert into a bag of words vector.
    total_features: int, the vocab size.

  Returns:
    bow_vector: tensor [vocab size], a bag of words representation of
      sparse_indices.
  """
  descending_indices, _ = tf.nn.top_k(sparse_indices, k=tf.size(sparse_indices))
  ascending_indices = tf.reverse(descending_indices, axis=[0])
  unique_indices, _ = tf.unique(ascending_indices)
  bow_vector = tf.sparse_to_dense(
      sparse_indices=unique_indices,
      output_shape=[total_features],
      sparse_values=1)

  return bow_vector
