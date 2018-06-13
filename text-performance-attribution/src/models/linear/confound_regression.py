"""Regression model which only has confound variables as features.

Given some text X, confounds C, and targets Y, this file
handles the logic for training a linear model which uses C as features and
predicts Y.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys; sys.path.append('../..')
import src.models.linear.regression_base as regression_base


class ConfoundRegression(regression_base.Regression):
  """A regression which only uses confounds as features."""

  def _iter_minibatches(self,
                        dataset,
                        target_name=None,
                        features=None,
                        level=None,
                        batch_size=None):
    """Continuously loops over the `dataset` and yields (confound, Y) pairs.

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
      x: np.array(int) [batch size, num features], confound occurrence vectors
        for a batch of examples. If example i has belongs to confound class j
        then x[i, j] == 1.
      y: np.array(int or float) [batch size], target labels for x.
        If the target class is C, then y[i] == 1 if example i belongs to C.
      x_features: list(string), column names for x, i.e. an ordered
        list of feature names.
    """

    i = 0
    while True:
      start = i
      end = (i + batch_size if batch_size else None)

      # If target_name is missing, we are doing inference so y can be None.
      if target_name is not None:
        y = dataset.y_batch(target_name, level, start, end)
      else:
        y = None

      x_confounds, confound_features = dataset.nontext_x_batch(
          self.confound_names, start, end)

      yield x_confounds, y, confound_features

      if batch_size is None:
        break

      i += batch_size
      if i + batch_size > dataset.split_sizes[dataset.current_split]:
        i = 0
