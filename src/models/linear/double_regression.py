"""
f(C, D) => y
g(T) => residual

"""

import sys
sys.path.append('../..')
import regression_base
import plain_regression

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


class DoubleRegression(plain_regression.RegularizedRegression):
    """ residualized regression """

    def __init__(self, config, params):
        plain_regression.RegularizedRegression.__init__(self, config, params)
        self.lmbda = self.params.get('lambda', 0)
        self.regularizor = self.params['regularizor'] if self.lmbda > 0 else None
        # to be filled up between passes
        # NOTE: if this isn't none, then we assume we are in the 2nd phase,
        #       where we are predicting residuals from text
        self.residuals = None 


    def _iter_minibatches(self, dataset, target_name=None, features=None, 
                                level=None, batch_size=None):
        i = 0
        while True:
            start = i
            end = (i+batch_size if batch_size else None)

            # we are in the 2nd pass, so get text covariates and y from residuals
            if self.residuals is not None:
                X, X_features = dataset.text_X_chunk(features, start, end)
                y = self.residuals[target_name][start:end]
                if level is not None:
                    target_col = dataset.class_to_id_map[target_name][level]
                    y = y[:, target_col]
                y = np.squeeze(np.asarray(y))  # TODO -- why isn't squeeze working properly?
                yield X, y, X_features

            # otherwise, yield confounds-only
            else:
                if target_name is not None:
                    y = dataset.y_chunk(target_name, level, start, end)
                else:
                    y = None
                X, X_features = dataset.nontext_X_chunk(
                    self.confound_names, start, end)

                yield X, y, X_features


    def _ovr_regression(self, dataset, target, features=None):
        """ run ovr-style regression (in lieu of classification)
        """
        models = {}
        for level in dataset.class_to_id_map[target['name']].keys():
            models[level] = self._fit_regression(
                dataset, target, level=level, features=features)
        return models


    def train_model(self, dataset, features=None):
        """ train a bunch of models and RETURN the nested dict
            instead of setting self.models
        """
        models = {}

        for i, target in enumerate(self.targets):
            fitting_kwargs = {'dataset': dataset, 'target': target, 'features': features}
            # fit a regression
            if target['type'] == 'continuous':
                models[target['name']] = plain_regression.RegularizedRegression._fit_regression(
                    self, **fitting_kwargs)
            # predict the residuals for each class if categorical 
            elif self.residuals is not None:
                models[target['name']] = self._ovr_regression(
                    **fitting_kwargs)
            # just fit a classification
            else:
                models[target['name']] = plain_regression.RegularizedRegression._fit_ovr(
                    self, **fitting_kwargs)
        return models

    def train(self, dataset, model_dir):
        # first train a model using the confounds only
        print "DOUBLE REGRESSION: first pass using confounds..."
        f = self.train_model(dataset)
        self.models = f
        start = time.time()

        # then get the residuals
        print "DOUBLE REGRESSION: inference for residuals..."
        preds = self.inference(dataset, model_dir).scores
        print "\tDone. Took %.2fs" % (time.time() - start)
        self.residuals = {}
        for i, target in enumerate(self.targets):
            y_hat = preds[target['name']]
            y = dataset.np_data[dataset.split][target['name']].toarray()
            self.residuals[target['name']] = y - y_hat

        # now predict the residuals using the text
        print "DOUBLE REGRESSION: 2nd pass using text and residuals..."
        g = self.train_model(dataset)
        self.models = g

