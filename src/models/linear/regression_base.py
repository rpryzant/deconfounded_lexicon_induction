""" base class for regressions """

import sys
sys.path.append('../..')

from collections import defaultdict, namedtuple
from src.models.abstract_model import Model, Prediction
import src.msc.utils as utils
import src.msc.utils as utils
import math
import pickle
import os
import numpy as np
import time
from functools import partial
from scipy import sparse



ModelResult = namedtuple('ModelResult', 
    ('model', 'response_type', 'weights'))


class Regression(Model):
    """ base class for all regression-type models

    """
    def __init__(self, config, params, intercept=True):
        Model.__init__(self, config, params)
        # target variable name (exploded if categorical)
        #     maps to ===>  R object with this model  
        self.models = {}
        self.use_intercept = intercept

        variables = [v for v in self.config.data_spec[1:] \
                        if not v.get('skip', False)]
        self.targets = [
            variable for variable in variables \
            if variable['control'] == False and not variable['skip']]
        self.confounds = [
            variable for variable in variables \
            if variable['control'] and not variable['skip']]
        self.confound_names = [
            variable['name'] for variable in self.confounds]


    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        models_file = os.path.join(model_dir, 'models')
        utils.pickle(self.models, models_file)
        print 'REGRESSION: models saved into ', models_file


    def load(self, dataset, model_dir):
        start = time.time()
        self.models = utils.depickle(os.path.join(model_dir, 'models'))
        target_names = map(lambda x: x['name'], self.targets)
        assert set(target_names) == set(self.models.keys())
        print 'REGRESSION: loaded model parameters from %s, time %.2fs' % (
            model_dir, time.time() - start)


    def _summarize_model_weights(self):
        def nested_model_iter(d):
            for _, model in d.iteritems():
                if isinstance(model, dict):
                    for k, v in nested_model_iter(model):
                        yield k, v
                else:
                    for k, v in model.weights.items():
                        yield k, v

        weights = defaultdict(list)
        # get duplicate of self.models except lists of weights
        for feature, value in nested_model_iter(self.models):
            weights[feature].append(value)
        out = {
            f: np.mean(v) for f, v in weights.iteritems()
        }
        return out


    def inference(self, dataset, model_dir):
        X, _, features = next(self._iter_minibatches(dataset))

        predictions = defaultdict(dict)
        for response_name, val in self.models.iteritems():
            if isinstance(val, dict):
                # convert {level: scores} to 2d matrix with rows like:
                #  level1 score, level2 score, etc
                # (where ordering is determined by the dataset)
                response_levels = dataset.num_levels(response_name)
                level_predictions = \
                    lambda level: self._predict(X, features, val[dataset.id_to_class_map[response_name][level]])
                arr = np.array(
                    [level_predictions(l) for l in range(response_levels)])
                if len(arr.shape) > 2:
                    arr = np.squeeze(arr, axis=2)
                predictions[response_name] = np.transpose(arr, [1, 0])
            else:
                predictions[response_name] = self._predict(X, features, val)

        average_coefs = self._summarize_model_weights()

        return Prediction(
            scores=predictions,
            feature_importance=average_coefs)


    def _predict(self, X, feature_names, model):
        def score(example):
            s = 0
            for xi, feature in zip(example, feature_names):
                s += model.weights.get(feature, 0) * xi
            s += (model.weights['intercept'] if self.use_intercept else 0)
            return s

        out = []
        for row in X:
            s = score(np.squeeze(row))
            if model.response_type == 'continuous':
                out.append(s)
            else:
                try:
                    out.append(1.0 / (1 + math.exp(-s)))
                except OverflowError:
                    out.append(1.0 if s > 0 else 0)
        return out


    def _fit_regression(self, dataset, target, level=None, features=None):
        raise NotImplementedError


    def _fit_classifier(self, dataset, target, level='', features=None):
        raise NotImplementedError



    def _fit_ovr(self, dataset, target, features=None):
        models = {}
        # class_to_id is a nested dict, 
        # each key (each categorical var) points to a dict mapping to ids
        # looping through all the possible classes of this categorical var
        for level in dataset.class_to_id_map[target['name']].keys():
            models[level] = self._fit_classifier(
                dataset, target, level=level, features=features)
        return models


    def train(self, dataset, model_dir, features=None):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report
        """
        for i, target in enumerate(self.targets):
            if target['type'] == 'continuous':
                self.models[target['name']] = self._fit_regression(
                    dataset=dataset, 
                    target=target,
                    features=features)
            else:
                self.models[target['name']] = self._fit_ovr(
                    dataset=dataset, 
                    target=target,
                    features=features)



