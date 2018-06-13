"""Base class for all linear models.

Subclasses must implement their own _fit_regression, _fit_classifier, and
_iter_minibatches functions. Everything else (prediction, generating
model summaries, saving, loading, one-vs-rest training) is handled by this.


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import namedtuple
import math
import os
import time
import numpy as np
from sklearn import linear_model
from tqdm import tqdm

import sys; sys.path.append('../..')

from src.models.abstract_model import Model
from src.models.abstract_model import Prediction
import src.msc.utils as utils

# Singleton class for packaging the results of an individual regression or
# classification model. For ordinal variables with multiple levels, the system
# trains a separate regression per level.
# See: https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest
ModelResult = namedtuple('ModelResult', ('model', 'response_type', 'weights'))


class Regression(Model):
  """Base class for all linear models."""

  def __init__(self, config, params, intercept=True):
    """Initializes a Regression by unpacking the target and confound variables.

    Args:
      config: NamedTuple, a config.yaml file that's been parsed into an object.
      params: dict, the part of the config which has to do with this model.
        Note that this dict is a member of config.model_spec.
      intercept: bool, whether or not we should fit an intercept.
    """
    Model.__init__(self, config, params)
    # This dict maps variable names to the model for that variable.
    # If a variable is categorical, then instead of a model for a value,
    # the value is a nested dictionary which maps categorical levels
    # to the model for that level.
    self.models = {}
    self.use_intercept = intercept

    # Get all of the variables which are prediction targets, as well
    # as all of the variables which are confounders.
    variables = [
        v for v in self.config.data_spec[1:] if not v.get('skip', False)
    ]
    self.targets = [
        variable for variable in variables if not variable['control']
    ]
    self.confounds = [variable for variable in variables if variable['control']]
    self.confound_names = [variable['name'] for variable in self.confounds]

    self.lmbda = self.params.get('lambda', 0)
    self.regularizer = self.params['regularizer'] if self.lmbda > 0 else None

  def save(self, model_dir):
    """Saves all of the models in self.models into `model_dir`.

    The models are saved as serialized pickle objects.
    See: https://docs.python.org/3/library/pickle.html

    Args:
      model_dir: string, the directory to save into.
    """
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    models_file = os.path.join(model_dir, 'models')
    utils.pickle(self.models, models_file)
    print('REGRESSION: models saved into %s' % models_file)

  def load(self, dataset, model_dir):
    """Loads self.models from `model_dir`."""
    start = time.time()
    self.models = utils.depickle(os.path.join(model_dir, 'models'))
    target_names = [x['name'] for x in self.targets]
    assert set(target_names) == set(self.models.keys())
    print('REGRESSION: loaded model parameters from %s, time %.2fs' % (
                 model_dir,
                 time.time() - start))

  def _summarize_model_weights(self):
    """Gets a single "importance value" for each feature from self.models."""
    out = {}
    for variable_name, variable_result in self.models.items():
      # This means that the current variable is categorical, since
      # self.models[categorical variable] maps to a {level => ModelResult}
      # dictionary.
      if isinstance(variable_result, dict):
        for level_name, level_result in variable_result.items():
          if variable_name not in out:
            out[variable_name] = {}
          out[variable_name][level_name] = level_result.weights
      else:
        out[variable_name] = variable_result.weights

    return out

  def inference(self, dataset, model_dir):
    """Uses self.models to perform inference over a dataset.

    Args:
      dataset: src.data.dataset.Dataset, the dataset for performing inference.
      model_dir: string, unused, but possibly used by subclasses.

    Returns:
      A src.models.abstract_model.Prediction object.
    """
    print('REGRESSION: getting data for inference...')
    x, _, features = next(self._iter_minibatches(dataset))

    predictions = defaultdict(dict)
    for response_name, model in self.models.iteritems():
      if isinstance(model, dict):
        # Convert {level: scores} to 2d matrix with columns:
        #   level1 score, level2 score, etc
        # (where ordering is determined by the dataset).
        response_levels = dataset.num_levels(response_name)
        arr = np.array([
            self._predict(x, features,
                          model[dataset.id_to_class_map[response_name][level]])
            for level in range(response_levels)
        ])
        # Squeeze out empty dimensions.
        if len(arr.shape) > 2:
          arr = np.squeeze(arr, axis=2)
        predictions[response_name] = np.transpose(arr, [1, 0])

      else:
        predictions[response_name] = self._predict(x, features, model)

    average_coefs = self._summarize_model_weights()

    return Prediction(scores=predictions, feature_importance=average_coefs)

  def _predict(self, x, feature_names, model):
    """Uses a model to create predictions for a bunch of covariates X.

    We are not using sklearn's predict() function because feature_names
      might be a subset of x's columns, which is a case that sklearn
      does not support.

    Args:
      x: np array [n examples, n features], the covariates to be inputted to
        the model.
      feature_names: list(string), column names for X.
      model: an instance of sklearn.linear_model, the model we are using
        for inference.

    Returns:
      out: list(float) or list(list(float)), predictions for each `x`.
    """

    def score(example):
      s = 0
      for xi, feature in zip(example, feature_names):
        s += model.weights.get(feature, 0) * xi
      s += (model.weights['intercept'] if self.use_intercept else 0)
      return s

    out = []
    for row in tqdm(x):
      s = score(np.squeeze(row))
      if model.response_type == 'continuous':
        out.append(s)
      else:
        try:
          out.append(1.0 / (1 + math.exp(-s)))
        except OverflowError:
          out.append(1.0 if s > 0 else 0)
    return out

  def _fit_one_vs_rest(self, dataset, target, features=None):
    """Fits a classifier to each level of a categorical variable (`target`).

    See: https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest


    Args:
      dataset: dataset.Dataset, the data we are fitting.
      target: dict, a member of config.data_spec, the variable we are
        predicting.
      features: list(string), an optional subset of the features we should
        restrict the model to.

    Returns:
      models: dict(string => regression_base.ModelResult): a trained model
        per level of the target variable.
    """
    models = {}
    # class_to_id is a nested dict where
    # each key (each categorical var) points to a dict mapping to ids.
    # So we are looping through all the possible classes of this categorical
    # variable.
    for level in dataset.class_to_id_map[target['name']].keys():
      models[level] = self._fit_classifier(
          dataset, target, level=level, features=features)
    return models

  def train(self, dataset, model_dir, features=None):
    """Trains a model for each target."""
    for target in self.targets:
      if target['type'] == utils.CONTINUOUS:
        self.models[target['name']] = self._fit_regression(
            dataset=dataset, target=target, features=features)
      else:
        self.models[target['name']] = self._fit_one_vs_rest(
            dataset=dataset, target=target, features=features)

  def _iter_minibatches(self,
                        dataset,
                        target_name=None,
                        features=None,
                        level=None,
                        batch_size=None):
    """Continuously loops over the `dataset` and yields (covariate, Y) pairs.

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

      # If target_name is missing, we are doing inference so y can be None.
      if target_name is not None:
        y = dataset.y_batch(target_name, level, start, end)
      else:
        y = None

      x, x_features = dataset.text_x_batch(features, start, end)

      yield x, y, x_features

      # If batch_size is missing, we are returning the whole dataset so
      # no need to keep iterating.
      if batch_size is None:
        break

      i += batch_size
      if i + batch_size > dataset.split_sizes[dataset.current_split]:
        i = 0

  def _sklearn_weights(self, model, feature_names):
    """Gets a feature_name=>weight mapping for the model."""
    weights = {}
    for w, f in zip(np.squeeze(model.coef_), feature_names):
      weights[f] = w
    if self.use_intercept:
      weights['intercept'] = model.intercept_
    return weights

  def _fit_regression(self, dataset, target, level=None, features=None):
    """Fits a regression -- to be implemented by subclasses.

    This method updates self.model[target] with the trained model and does
    not return anything.

    Args:
      dataset: src.data.dataset.Dataset, the data which is to be used
        for fitting.
      target: string, the name of the target variable.
      level: string, the target's sub-class. If this isn't specified, the system
        will assume that the target is monolithic.
      features: list(string), a subset of dataset.vocab which is to be used
        while fitting.

    Returns:
      regression_base.ModelResult, the fitted parameters.
    """
    iterator = self._iter_minibatches(
        dataset=dataset,
        target_name=target['name'],
        features=features,
        batch_size=self.params['batch_size'],
        level=level)

    print('REGRESSION: fitting target %s', target['name'])
    model = linear_model.SGDRegressor(
        penalty=self.regularizer or 'none',
        alpha=self.lmbda,
        learning_rate='constant',
        eta0=self.params.get('lr', 0.001))

    for _ in tqdm(range(self.params['num_train_steps'])):
      xi, yi, x_features = next(iterator)
      model.partial_fit(xi, yi)

    return ModelResult(
        model=model,
        weights=self._sklearn_weights(model, x_features),
        response_type='continuous')

  def _fit_classifier(self, dataset, target, level=None, features=None):
    """Fits a classifier -- to be implemented by subclasses.

    Multiclass classification is done with OVR (one versus rest) classification.
    This means that there is a separate regression for each class, and
    each of these regressions is trained to pick this class out.

    This method updates self.model[target] with the trained model and does
    not return anything.

    Args:
      dataset: src.data.dataset.Dataset, the data to be used for fitting.
      target: string, the name of the target variable.
      level: string, the target's sub-class. If this isn't specified, the system
        will assume that the target is monolithic.
      features: list(string), a subset of dataset.vocab which is to be
        used while fitting.

    Returns:
      regression_base.ModelResult, the fitted parameters.
    """
    iterator = self._iter_minibatches(
        dataset=dataset,
        target_name=target['name'],
        features=features,
        level=level,
        batch_size=self.params['batch_size'])

    print('CLASSIFICATION: fitting target %s, level %s', target['name'],
                 level)
    model = linear_model.SGDClassifier(
        loss='log',
        penalty=(self.regularizer or 'none'),
        alpha=self.lmbda,
        learning_rate='constant',
        eta0=self.params.get('lr', 1.0))

    for _ in tqdm(range(self.params['num_train_steps'])):
      xi, yi, x_features = next(iterator)
      model.partial_fit(xi, yi, classes=[0., 1.])

    return ModelResult(
        model=model,
        weights=self._sklearn_weights(model, x_features),
        response_type='categorical')
