
import regression_base
from functools import partial

import sklearn
import numpy as np
from tqdm import tqdm
from scipy import sparse


class RegularizedRegression(regression_base.Regression):
    """ traditional regression """ 

    def __init__(self, config, params, intercept=True):
        regression_base.Regression.__init__(self, config, params, intercept)
        self.lmbda = self.params.get('lambda', 0)
        self.regularizor = self.params['regularizor'] if self.lmbda > 0 else None


    def _iter_minibatches(self, dataset, target_name=None, features=None, 
                                level=None, batch_size=None):
        i = 0
        while True:
            start = i
            end = (i+batch_size if batch_size else None)

            if target_name is not None:
                y = dataset.y_chunk(target_name, level, start, end)
            else:
                y = None

            X, X_features = dataset.text_X_chunk(features, start, end)
            yield X, y, X_features

            if batch_size is None:
                break

            i += batch_size
            if i + batch_size > dataset.split_sizes[dataset.split]:
                i = 0


    def _sklearn_weights(self, model, feature_names):
        weights = {}
        for w, f in zip(np.squeeze(model.coef_), feature_names):
            weights[f] = w
        if self.use_intercept:
            weights['intercept'] = model.intercept_
        return weights


    def _fit_regression(self, dataset, target, level=None, features=None):
        iterator = self._iter_minibatches(
            dataset=dataset,
            target_name=target['name'],
            features=features,
            batch_size=self.params['batch_size'],
            level=level)

        print 'REGRESSION: fitting target %s' % target['name']
        model = sklearn.linear_model.SGDRegressor(
            penalty=self.regularizor or 'none',
            alpha=self.lmbda,
            learning_rate='constant',
            eta0=self.params.get('lr', 0.001))

        for _ in tqdm(range(self.params['num_train_steps'])):
            Xi, yi, X_features = next(iterator)
            model.partial_fit(Xi, yi)

        return regression_base.ModelResult(
            model=model,
            weights=self._sklearn_weights(model, X_features),
            response_type='continuous')


    def _fit_classifier(self, dataset, target, level=None, features=None):
        iterator = self._iter_minibatches(
            dataset=dataset,
            target_name=target['name'],
            features=features,
            level=level,
            batch_size=self.params['batch_size'])

        print 'CLASSIFICATION: fitting target %s, level %s' % (target['name'], level)
        model = sklearn.linear_model.SGDClassifier(
            loss='log',
            penalty=(self.regularizor or 'none'),
            alpha=self.lmbda,
            learning_rate='constant',
            eta0=self.params.get('lr', 1.0))

        for _ in tqdm(range(self.params['num_train_steps'])):
            Xi, yi, X_features = next(iterator)
            model.partial_fit(Xi, yi, classes=[0., 1.])

        return regression_base.ModelResult(
            model=model,
            weights=self._sklearn_weights(model, X_features),
            response_type='categorical')

