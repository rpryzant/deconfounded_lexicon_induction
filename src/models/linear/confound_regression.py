
import regression_base
import plain_regression
from functools import partial

import sklearn
import pandas as pd
import numpy as np
from scipy import sparse



class ConfoundRegression(plain_regression.RegularizedRegression):
    """ regression which uses confounds as the only features """

    def _iter_minibatches(self, dataset, target_name=None, features=None, 
                                level=None, batch_size=None):
        plain_iterator = plain_regression.RegularizedRegression._iter_minibatches(
            self, dataset, target_name, features, level, batch_size)

        i = 0
        while True:
            start = i
            end = (i+batch_size if batch_size else None)

            _, y, _ = next(plain_iterator)

            X_confounds, confound_features = dataset.nontext_X_chunk(
                self.confound_names, start, end)

            yield X_confounds, y, confound_features

            if batch_size is None:
                break

            i += batch_size
            if i + batch_size > dataset.split_sizes[dataset.split]:
                i = 0



