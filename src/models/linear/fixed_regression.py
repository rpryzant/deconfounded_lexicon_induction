
import regression_base
import plain_regression
from functools import partial

import sklearn
import pandas as pd
import numpy as np
from scipy import sparse



class FixedRegression(plain_regression.RegularizedRegression):
    """ regression which includes confounds as features """

    def _iter_minibatches(self, dataset, target_name=None, features=None, 
                                level=None, batch_size=None):
        plain_iterator = plain_regression.RegularizedRegression._iter_minibatches(
            self, dataset, target_name, features, level, batch_size)

        i = 0
        while True:
            start = i
            end = (i+batch_size if batch_size else None)

            X_text, y, text_features = next(plain_iterator)

            X_confounds, confound_features = dataset.nontext_X_chunk(
                self.confound_names, start, end)

            X = np.column_stack([X_text, X_confounds])
            X_features = text_features + confound_features

            yield X, y, X_features

            if batch_size is None:
                break

            i += batch_size
            if i + batch_size > dataset.split_sizes[dataset.split]:
                i = 0



