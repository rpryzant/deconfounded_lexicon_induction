"""Inference clients that are in charge of testing models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import numpy as np

import sys; sys.path.append('../..')
import src.msc.utils as utils


class InferenceClient(object):
  """Inference clients that are in charge of testing models."""

  def bow_model_inference(self, sess, feature_weights_op, output_op):
    """Tests a bag-of-words style model on a batch of inputs.

    Args:
      sess: tf.Session, the current TensorFlow session.
      feature_weights_op: tensor [vocab size, config.encoder_layers],
        the parameters which multiply the bag-of-words text vector input. Note
        that the 0th axis of this tensor is expected to correspond to the same
        ordering as the vocab IDs in the dataset. So for example if "the" has
        ID 3 in self.dataset, the parameters at feature_weights_op[3] should
        be multiplying "the" in the model.
      output_op: dict(string => dict(string => tensor)), a mapping with the
        following form:
        {
          "variable_name": {
            "loss": tf.float32, the model's average loss on this variable (l2
              if continuous and cross-entropy if categorical)
            "input": tensor [batch size] (tf.int32 or tf.float32), the ground-
              truth labels for this variable. If the variable is categorical
              then these labels are the IDs of the correct class for each
              example.
            "pred": tensor [batch size] or [batch size, num_classes],
              predictions or logits for each example.
            "weights": tensor [hidden + confunds, num outputs], the final layer
              that combines the encodings + pre-predictions together for the
              final predictions.
            }
        }

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
    input_weights, output = sess.run(
        [feature_weights_op, output_op], feed_dict={self.dropout: 0.0})

    predictions = defaultdict(list)
    token_importance = defaultdict(dict)

    # Get predictions and token importance values for each variable.
    for variable_name in output:

      # Remember the predictions on this variable.
      predictions[variable_name] += output[variable_name]['pred'].tolist()

      for level_name, level_id in self.dataset.class_to_id_map[
          variable_name].items():
        importance_dict = defaultdict(list)

        for feature_id, feature_name in enumerate(
            self.dataset.ordered_features):
          for idx in range(self.params['encoding_dim']):
            input_weight = input_weights[feature_id, idx]
            output_weight = output[variable_name]['weights'][idx, level_id]
            importance_dict[feature_name].append(input_weight * output_weight)

        token_importance[variable_name][level_name] = importance_dict

    return predictions, token_importance

  def attn_model_inference(self, sess, input_text_op, output_op,
                           attn_scores_op):
    """Tests an LSTM/attention-style model on a batch of inputs.

    Args:
      sess: tf.Session, the current TensorFlow session.
      input_text_op: tensor [batch_size, max seq length] tf.string, the
        textual input sequences to the model.
      output_op: dict(string => dict(string => tensor)), a mapping with the
        following form:
        {
          "variable_name": {
            "loss": tf.float32, the model's average loss on this variable (l2
              if continuous and cross-entropy if categorical)
            "input": tensor [batch size] (tf.int32 or tf.float32), the ground-
              truth labels for this variable. If the variable is categorical
              then these labels are the IDs of the correct class for each
              example.
            "pred": tensor [batch size] or [batch size, num_classes],
              predictions or logits for each example.
            }
        }
      attn_scores_op: tensor [batch size, max seq len] tf.float32, the
        attentional distributions over each input sequence.

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

    def add_token_scores(token_seq, scores, importance_dict):
      """Helper that maps tokens to their scores given a (non-default)dict."""
      for token, score in zip(token_seq, scores):
        # Skip OOV tokens.
        if token not in self.dataset.features:
          continue
        # We can't use a defaultdict here because importance_dict is nested
        # within a larger dictionary and the nesting is not consistent.
        # For continuous variables we are mapping straight from variable name to
        # this importance_dict. For categorical variables we are mapping
        # from variable name to variable level, and then to this importance
        # dict.
        if token not in importance_dict:
          importance_dict[token] = [score]
        else:
          importance_dict[token].append(score)

    ops = [input_text_op, output_op, attn_scores_op]
    inputs, outputs, scores = sess.run(ops, feed_dict={self.dropout: 0.0})

    predictions = defaultdict(list)
    token_importance = defaultdict(dict)

    # Remember the predictions and feature importance values for each variable.
    for variable_name in outputs:
      variable = self.dataset.get_variable(variable_name)

      # Remember the predictions on this variable.
      predictions[variable_name] += outputs[variable_name]['pred'].tolist()

      # For continuous variables we map from variable names to a
      # {token: importance value} dictionary.
      if variable['type'] == utils.CONTINUOUS:
        for input_seq, attn_distribution in zip(inputs, scores):
          add_token_scores(input_seq, attn_distribution,
                           token_importance[variable_name])

      # For categorical variables we map from variable names to variable
      # levels and then to a {token: importance value} dictionary.
      else:
        labels = outputs[variable_name]['input']
        for label, input_seq, attn_distribution, logits in zip(
            labels, inputs, scores, predictions[variable_name]):

          # Only record true positives for the target variable.
          if (variable_name == self.config.eval_variable_name and
              label != np.argmax(logits)):
            continue

          level_name = self.dataset.id_to_class_map[variable_name][label]

          if level_name not in token_importance[variable_name]:
            token_importance[variable_name][level_name] = {}
          add_token_scores(input_seq, attn_distribution,
                           token_importance[variable_name][level_name])

    return predictions, token_importance

  def cnn_model_inference(self, sess, input_text_op, output_op):
    """Tests a CNN-style model on a batch of inputs.

    Note that the CNN will capture ngrams, so the data are expected to be
    tokenized as such.

    Args:
      sess: tf.Session, the current TensorFlow session.
      input_text_op: tensor [batch_size, max seq length] tf.string, the
        textual input sequences to the model.
      output_op: dict(string => dict(string => tensor)), a mapping with the
        following form:
        {
          "variable_name": {
            "loss": tf.float32, the model's average loss on this variable (l2
              if continuous and cross-entropy if categorical)
            "input": tensor [batch size] (tf.int32 or tf.float32), the ground-
              truth labels for this variable. If the variable is categorical
              then these labels are the IDs of the correct class for each
              example.
            "pred": tensor [batch size] or [batch size, num_classes],
              predictions or logits for each example.
            "conv": list(tensor [batch size, seq len - (filter width - 1),
                                                                  n filters])
              the feature maps which were fed into the final fc/softmax layer
              for each filter width.
            "weights": tensor  [n filters * n filter widths, n classes] the
              parameters of the fc layer that is multiplied by an average-pooled
              version of "conv" to produce softmax inputs.
            }
        }

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

    def get_cams_for_batch(feature_map, weights, preds, labels, outcome_id):
      """Get the class activation maps for a batch of feature maps."""
      # Get a weighted average of feature maps.
      batch_cams = np.zeros((feature_map.shape[:2]))
      for filter_idx in range(self.params['n_filters']):
        batch_cams += feature_map[:, :, filter_idx] * weights[filter_idx,
                                                              outcome_id]

      # Subset the cams to only include maps from true positives.
      if variable['type'] == utils.CATEGORICAL and self.params['tp_only']:
        true_positive_mask = np.logical_and(
            np.argmax(preds, axis=1) == labels, labels == outcome_id)
        return batch_cams[true_positive_mask], text_inputs[true_positive_mask]
      return batch_cams, text_inputs

    ops = [input_text_op, output_op]
    text_inputs, step_outputs = sess.run(ops, feed_dict={self.dropout: 0.0})
    predictions = defaultdict(list)
    token_importance = defaultdict(dict)

    for variable_name in step_outputs:
      variable = self.dataset.get_variable(variable_name)

      if variable['control'] or variable['skip']:
        continue

      # Pull out everything we need from the step outputs.
      variable_labels = step_outputs[variable_name]['input']
      feature_maps = step_outputs[variable_name]['conv']
      final_fc_weights = step_outputs[variable_name]['weights']
      variable_preds = step_outputs[variable_name]['pred']

      # Remember the predictions on this variable.
      predictions[variable_name] += variable_preds.tolist()

      # Get the token scores for each ngram size (= filter width).
      for i, (feature_map, filter_width) in enumerate(
          zip(feature_maps, self.filter_sizes)):
        # These are the weights which correspond to the current filter width.
        chunk_start = i * self.params['n_filters']
        weight_chunk = final_fc_weights[chunk_start:chunk_start +
                                        self.params['n_filters']]

        # Get the number of horizontal filter applications.
        num_filter_applications = feature_map.shape[1]

        # Get the Class Activation Map (CAM) for each level of the current
        # outcome variable. A CAM is a weighted combination of feature maps,
        # where the weights come from `weight_chunk`.
        for outcome_id in range(weight_chunk.shape[-1]):
          batch_cams, input_seqs = get_cams_for_batch(
              feature_map, weight_chunk, variable_preds, variable_labels,
              outcome_id)

          # Now that we have cams for all of the true positives, pull out all of
          # the ngram scores and then add these to the outgoing dict.
          ngram_score_map = defaultdict(list)
          for token_seq, cam in zip(input_seqs, batch_cams):
            for timestep, score in zip(range(num_filter_applications), cam):
              ngram = '_'.join(token_seq[timestep:timestep + filter_width])

              if ngram not in self.dataset.features:
                continue

              ngram_score_map[ngram].append(score)

          # Merge the current token scores with anything that's already
          # in the outgoing dictionary (there may already be token scores
          # from another ngram type).
          if variable['type'] == utils.CONTINUOUS:
            if variable_name not in token_importance:
              token_importance[variable_name] = ngram_score_map
            else:
              token_importance[
                  variable_name] = dict(ngram_score_map.items() +
                                        token_importance[variable_name].items())
          else:
            level_name = self.dataset.id_to_class_map[variable_name][outcome_id]
            if level_name not in token_importance[variable_name]:
              token_importance[variable_name][level_name] = ngram_score_map
            else:
              token_importance[variable_name][level_name] = dict(
                  token_importance[variable_name][level_name].items() +
                  ngram_score_map.items())

    return predictions, token_importance
