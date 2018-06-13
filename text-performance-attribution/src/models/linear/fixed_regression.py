"""A regression which uses covariates + confounds to predict outcomes.

Given some text X, confounds C, and target outcomes Y, this file contains
code for training a regression that predicts Y using X and C as features.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import sys; sys.path.append('../..')
import src.models.linear.regression_base as regression_base


class FixedRegression(regression_base.Regression):
  """A regression which uses covariates + confounds to predict outcomes."""

  def _iter_minibatches(self,
                        dataset,
                        target_name=None,
                        features=None,
                        level=None,
                        batch_size=None):
    """Continuously loops over the data, yielding (confound+covariate, Y) pairs.

    If batch_size is None then we iterate once. Otherwise the generator
    will continuously cycle over the data.

    Args:
      dataset: src.data.dataset.Dataset, the dataset we are iterative over.
      target_name: string, the name of the variable that should be used
        for the targets (Y).
      features: list(string), a subset of the features that we should select
        when pulling X from the data. If this isn't provided, then X will
        include all features in the data.
      level: string, the categorical level which is to be retrieved for Y.
        If supplied, Y is assumed to be categorical.
      batch_size: int, the batch size to use.

    Yields:
      x: np.array(int or float) [batch size, num features], occurrence vectors
        for a batch of examples. If example i has feature j then x[i, j] == 1.
        The features in X can be integer text features *and* confound features.
        which may be continuous.
      y: np.array(int or float) [batch size], target labels for x.
        If the target class is C, then y[i] == 1 if example i belongs to C.
      x_features: list(string), column names for x, i.e. an ordered
        list of feature names.
    """
    plain_iterator = regression_base.Regression._iter_minibatches(
        self, dataset, target_name, features, level, batch_size)

    i = 0
    while True:
      start = i
      end = (i + batch_size if batch_size else None)
      # First, call the plain iterator of the superclass to get X, Y pairs.
      x_text, y, text_features = next(plain_iterator)

      # Now get confounds for the same batch.
      x_confounds, confound_features = dataset.nontext_x_batch(
          self.confound_names, start, end)

      # Then attach the confounds to the covariates.
      x = np.column_stack([x_text, x_confounds])
      x_features = text_features + confound_features

      yield x, y, x_features

      if batch_size is None:
        break

      i += batch_size
      if i + batch_size > dataset.split_sizes[dataset.current_split]:
        i = 0
