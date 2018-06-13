"""A pair of regressions where the 2nd predicts is predecessors residuals.

Given some text X, confounds C, and target outcomes Y, this class will train
a "residualized regression" in the style of
https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=1022.

This means training a pair of regressions. The first uses C to predict Y.
The second regression uses C to predict the residuals of the first regression
(i.e., (Y - Y_hat)).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import sys; sys.path.append('../..')
import src.models.linear.regression_base as regression_base
import src.msc.utils as utils


class DoubleRegression(regression_base.Regression):
  """First, one regression uses the confounds to predict the target.

  Second, another regression uses the covariates to predict the residuals
  of the first model.
  """

  def __init__(self, config, params):
    regression_base.Regression.__init__(self, config, params)
    self.lmbda = self.params.get('lambda', 0)
    self.regularizer = self.params['regularizer'] if self.lmbda > 0 else None
    # self.residuals is to be filled up between passes, and will hold
    # the residuals (i.e., errors) of the first model.
    # NOTE: if this isn't none, then we assume we are in the 2nd phase,
    #       where we are predicting residuals from text.
    self.residuals = None

  def _iter_minibatches(self,
                        dataset,
                        target_name=None,
                        features=None,
                        level=None,
                        batch_size=None):
    """Continuously loops over the `dataset` and yields data pairs.

    The data pairs can come in two forms, depending on what stage of training
    the model is in. If the model is on the first training pass, then the
    pairs will be (confound, Y). If the model is on the second pass then the
    pairs will be (text features, (Y - Y_hat) ).

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
      x: np.array(int) [batch size, num features], word occurrence vectors
        for a batch of examples. If example i has feature j then x[i, j] == 1.
      y: np.array(int or float) [batch size], target labels for x.
        If the target class is C, then y[i] == 1 if example i belongs to C.
      x_features: list(string), column names for x, i.e. an ordered
        list of feature names.
    """
    i = 0
    while True:
      start = i
      end = (i + batch_size if batch_size else None)

      # We are in the 2nd pass, so get text covariates and y from residuals.
      if self.residuals is not None:
        x, x_features = dataset.text_x_batch(features, start, end)
        y = self.residuals[target_name][start:end]
        # Extract the categorical level we care about.
        if level is not None:
          target_col = dataset.class_to_id_map[target_name][level]
          y = y[:, target_col]
        y = np.squeeze(np.asarray(y))

        yield x, y, x_features

      # We are in the first pass, so yield confounds and traditional y's.
      else:
        if target_name is not None:
          y = dataset.y_batch(target_name, level, start, end)
        else:
          y = None
        x, x_features = dataset.nontext_x_batch(self.confound_names, start, end)

        yield x, y, x_features

  def _one_vs_rest_regression(self, dataset, target, features=None):
    """Fits several regressions, one per level of a target variable."""
    models = {}
    for level in dataset.class_to_id_map[target['name']].keys():
      models[level] = self._fit_regression(
          dataset, target, level=level, features=features)
    return models

  def train_model(self, dataset, features=None):
    """Trains some models and returns them instead of setting self.models."""
    models = {}

    for target in self.targets:
      fitting_kwargs = {
          'dataset': dataset,
          'target': target,
          'features': features
      }
      # Fit a regression.
      if target['type'] == utils.CONTINUOUS:
        models[target['name']] = regression_base.Regression._fit_regression(
            self, **fitting_kwargs)
      # Predict the residuals for each class if it's a categorical variable.
      elif self.residuals is not None:
        models[target['name']] = self._one_vs_rest_regression(**fitting_kwargs)
      # If its a categorical variable and we have residuals,
      # then predict that variable.
      else:
        models[target['name']] = regression_base.Regression._fit_one_vs_rest(
            self, **fitting_kwargs)
    return models

  def train(self, dataset, model_dir):
    """Residualization training procedure."""
    # First train a model using the confounds only.
    print('DOUBLE REGRESSION: first pass using confounds...')
    f = self.train_model(dataset)
    self.models = f
    start = time.time()

    # Then get the residuals.
    print('DOUBLE REGRESSION: inference for residuals...')
    preds = self.inference(dataset, model_dir).scores
    print('\tDone. Took %.2fs' % (time.time() - start))
    self.residuals = {}
    for target in self.targets:
      y_hat = preds[target['name']]
      y = dataset.np_data[dataset.current_split][target['name']].toarray()
      self.residuals[target['name']] = y - y_hat

    # Finally, predict the residuals using the text.
    # Because self.residuals is no longer None, the system will automatically
    # use these as regression/classification targets instead of the true
    # targets inside of the `dataset`.
    print('DOUBLE REGRESSION: 2nd pass using text and residuals...')
    g = self.train_model(dataset)
    self.models = g
