"""Utility functions for computing feature association statistics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import sys; sys.path.append('../..')
import src.msc.utils as utils

np.warnings.filterwarnings('ignore')


def log_odds(word, outcome_variable, dataset, config):
  """Computes log-odds between a word and each level of a categorical outcome.

  Log-odds = log pi - log (1 - pi)
    Where pi = the probability of `word` occurring in examples belonging to
    level i of the `outcome_variable`.

  Args:
    word: string, the word we are interested in.
    outcome_variable: string, the name of the outcome variable we are
      interested in. This MUST be a categorical variable.
    dataset: src.data.dataset.Dataset object, the dataset we are computing over.
    config: a configuration object.
  Returns:
    out: dict(string => float), a mapping from categorical level names
      to the log-odds between `word` and that particular level.
  """
  # Get a binary matrix where the rows are bag-of-words representations of each
  # input sequence and the columns are individual words.
  all_word_occurences = dataset.np_data[config.train_suffix][
      dataset.input_varname()].toarray()
  # Pull out the column that corresponds to the `word` of interest.
  # This is a 0/1 vector where a 1 at index i means the word occurred in example
  # i.
  selected_word_occurances = all_word_occurences[:, dataset.features[word]]

  out = {}
  one_hots = dataset.np_data[config.train_suffix][outcome_variable].toarray()
  for level_name, level_id in dataset.class_to_id_map[outcome_variable].items():
    # Get a 0/1 vector where a 1 at index i means example i belongs to the
    # current class.
    level_mask = one_hots[:, level_id]

    # Get probability (num within-class occurances / total occurances).
    prob_occurrence = np.sum(selected_word_occurances *
                             level_mask) / np.sum(selected_word_occurances)\

    # If the word doesn't occur in the data at all, then we will be dividing
    # by zero at line 49. Instead of keeping the nan we say that the word
    # isn't informative at all (i.e. that it has a probability of 0.5).
    if np.isnan(prob_occurrence):
      prob_occurrence = 0.5
    if not prob_occurrence:
      prob_occurrence += 1e-3
    elif prob_occurrence == 1:
      prob_occurrence -= 1e-3

    out[level_name] = math.log(prob_occurrence) - math.log(1 - prob_occurrence)

  return out


def cramers_v(feature, text, targets, possible_labels):
  """Computes the association strength between a word and a categorical outcome.

     See: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

  Args:
    feature: string, the word which is to be computed for.
    text: list(list(string)), the provided corpus.
    targets: list(string), labels for each sequence in `text`.
    possible_labels: list(string), the set of possible elements in `labels`.

  Returns:
    V: int, the chisq statistic for a single feature, given some text
       and target info (Y) and possible_labels (possible values for Y).
  """
  num_rows = 2
  num_cols = len(possible_labels)
  obs = np.zeros((num_rows, num_cols))

  for description, target in zip(text, targets):
    if feature in description:
      obs[1, possible_labels.index(target)] += 1
    else:
      obs[0, possible_labels.index(target)] += 1

  row_totals = np.sum(obs, axis=1)
  col_totals = np.sum(obs, axis=0)
  n = np.sum(obs)
  expected = np.outer(row_totals, col_totals) / n
  chisq = np.sum(np.nan_to_num(((obs - expected)**2) / expected))

  phisq = chisq / n
  v = np.sqrt(phisq / min(num_cols - 1, num_rows - 1))
  return v


def pointwise_biserial(feature, text, targets):
  """Computes the association strength between a word and a continuous outcome.

  See: https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient

  Args:
    feature: string, the word in question.
    text: list(list(string)), the corpus which is to be computed over.
    targets: list(float), the continuous outcome for each corpus sequence.
  Returns:
    rpb: float, the point biserial coefficient between `feature` and `text`.
  """
  group0 = []
  group1 = []
  for text_example, val in zip(text, targets):
    if feature in text_example:
      group1.append(val)
    else:
      group0.append(val)

  m0 = np.mean(group0) if group0 else 0.0
  m1 = np.mean(group1) if group1 else 0.0

  n0 = float(len(group0))
  n1 = float(len(group1))
  n = n0 + n1

  # Guard against division-by-zero.
  s = np.std(targets) or 1e-3

  rpb = ((m1 - m0) / s) * np.sqrt((n0 * n1) / (n**2))
  return rpb


def compute_correlation(var, features, input_text, labels):
  """Compute the average correlation between a list of features and labels.

  Args:
    var: dict, the variable which is to be analyzed. This is an element of
      config.data_spec.
    features: list(string), a list of words to compute correlations for.
    input_text: list(list(word)), a list of word sequences which is the
      corpus to be computed over.
    labels: list(string/float), a list of labels for each sequence in
      `input_text`. These can be strings (for categorical variables) or floats.
  Returns:
    A float, the average strength of association between a subset of words in
      `input_text` (where this subset is described by `features`) and
      the provided labels.
  """
  if var['type'] == utils.CATEGORICAL:
    return np.mean([
        cramers_v(
            feature=f,
            text=input_text,
            targets=labels,
            possible_labels=list(set(labels))) for f in features
    ])
  else:
    return np.mean([
        pointwise_biserial(feature=f, text=input_text, targets=labels)
        for f in features
    ])
