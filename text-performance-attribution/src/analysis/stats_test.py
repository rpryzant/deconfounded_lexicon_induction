"""Tests for creative_text_attribution.src.analysis.correlations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.sparse import csr_matrix
import pytest
import unittest
import path; sys.path.append('../..')
from src.analysis import stats


class DummyDataset(object):
  """A dummy dataset.Dataset object for test_log_odds."""

  def __init__(self):
    # This is a mapping from split_name => variable_name => data for that
    # variable.
    self.np_data = {
        'train': {
            # Categorical variables have their levels on the columns. The
            # column names are identified with the key-value pairs of
            # self.class_to_id_map.
            'categorical_var': csr_matrix(np.array([[0, 1], [1, 0], [1, 0]])),
            # Each row is a bag-of-words representation of some input sentence.
            # The columns are identified with the key-value pairs of
            # self.features (keys = col name, values = col index).
            # So for example, row 0 in this matrix corresponds to the sequence,
            # "tok0 tok1".
            'input_var': csr_matrix(
                np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]]))
        }
    }
    # Column names for the `categorical_var` data matrix in self.np_data.
    self.class_to_id_map = {'categorical_var': {'level_a': 0, 'level_b': 1}}
    # Column names for the `input_var` data matrix in self.np_data.
    self.features = {'tok0': 0, 'tok1': 1, 'tok2': 2}

  def input_varname(self):
    """Returns the name of the variable that is the text input."""
    return 'input_var'


class DummyConfig(object):
  """A dummy yaml config object for test_log_odds."""

  def __init__(self):
    self.train_suffix = 'train'


class StatsTest(unittest.TestCase):

  def within_epsilon(self, number, target, epsilon):
    """Tests whether a `number` is within `epsilon` of a `target`.

    Args:
      number: float, the number we are testing.
      target: float, what we are comparing `number` too.
      epsilon: float, the acceptable range we are willing to consider.

    Returns:
      True if (`number` - `target`) < `epsilon`.
    """
    return (number - target) < epsilon

  def xtest_compute_correlation_continuous_outcomes(self):
    """Tests correlation-computation between words and continuous outcomes."""
    variable = {'type': 'continuous'}
    input_text = [
        'i want to get our team a trophy .'.split(),
        'because our correlation is above average .'.split(),
    ]
    labels = [100.2, -3.1]

    # "features" has a strong positive association with "labels" because the
    # only input with a high label contains the selected word (`get`).
    features = ['get']
    correlation = stats.compute_correlation(variable, features, input_text,
                                            labels)
    self.assertTrue(self.within_epsilon(correlation, 1.0, 1e-3))

    # "features" has a strong negative association with "labels" because the
    # only inputs with low labels contains the selected word (`is`).
    features = ['is']
    correlation = stats.compute_correlation(variable, features, input_text,
                                            labels)
    self.assertTrue(self.within_epsilon(correlation, -1.0, 1e-3))

    # Make sure we are getting the **average** correlation between
    # elements in `features` and `labels`.
    features = ['get', 'is']
    correlation = stats.compute_correlation(variable, features, input_text,
                                            labels)
    self.assertTrue(self.within_epsilon(correlation, 0.0, 1e-3))

    # "features" is not correlated with the outcome because occurs in
    # both high and low examples.
    features = ['our']
    correlation = stats.compute_correlation(variable, features, input_text,
                                            labels)
    self.assertTrue(self.within_epsilon(correlation, 0.0, 1e-3))

  def test_compute_correlation_categorical_outcomes(self):
    """Tests correlation-computation between words and categorical outcomes."""
    variable = {'type': 'categorical'}
    input_text = [
        'i want to get our team a trophy .'.split(),
        'because our correlation is above average .'.split(),
    ]
    labels = [1, 0]

    # "feature" has a strong positive association with labels because
    # `get` only occurs in examples with the label `1`. So the presence of `get`
    # gives us lots of information as to what class the example is.
    features = ['get']
    correlation = stats.compute_correlation(variable, features, input_text,
                                            labels)
    self.assertTrue(self.within_epsilon(correlation, 1.0, 1e-3))

    # "features" still has a strong positive association with labels,
    # because `is` only co-occurs with the label `0`. So the presence of `is`
    # gives us a lot of information as to what class the example is.
    # While scalar correlations are directed, binary categorical correlations
    # are undirected.
    # See: https://stats.stackexchange.com/questions/256344/
    features = ['is']
    correlation = stats.compute_correlation(variable, features, input_text,
                                            labels)
    self.assertTrue(self.within_epsilon(correlation, 1.0, 1e-3))

    # "features" is unrelated to the labels.
    features = ['our']
    correlation = stats.compute_correlation(variable, features, input_text,
                                            labels)
    self.assertTrue(self.within_epsilon(correlation, 0.0, 1e-3))

    # We are computing the average feature correlation.
    features = ['get', 'is', 'our']
    correlation = stats.compute_correlation(variable, features, input_text,
                                            labels)
    self.assertTrue(self.within_epsilon(correlation, 0.6666, 1e-3))

  def test_log_odds(self):
    """Tests log-ods between words and categorical outcomes."""
    dataset = DummyDataset()
    config = DummyConfig()
    outcome_variable = 'categorical_var'

    # The first token (`tok0`) only occurs in the second class (`level_b`).
    word = 'tok0'
    results = stats.log_odds(word, outcome_variable, dataset, config)
    self.assertGreater(results['level_b'], results['level_a'])
    self.assertGreater(results['level_b'], 0)
    self.assertGreater(0, results['level_a'])

    # The second token occurs once in each class.
    word = 'tok1'
    results = stats.log_odds(word, outcome_variable, dataset, config)
    self.assertEqual(results['level_b'], results['level_a'])
    self.assertEqual(results['level_b'], 0)

    # The third token occurs twice in class a but only once in class b.
    word = 'tok2'
    results = stats.log_odds(word, outcome_variable, dataset, config)
    # log(0.33) - log(1 - 0.33)
    expected_a = 0.693
    self.assertLess(abs(results['level_a'] - expected_a), 0.01)
    # log(0.66) - log(1 - 0.66)
    expected_b = -0.693
    self.assertLess(abs(results['level_b'] - expected_b), 0.01)



