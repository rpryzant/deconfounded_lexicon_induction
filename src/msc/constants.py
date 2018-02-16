"""
classloader for models
"""

import sys
sys.path.append('../..')

from src.models.neural.tf_wrapper import ABOWWrapper
from src.models.neural.tf_wrapper import AATTNWrapper
from src.models.neural.tf_wrapper import DRBOWWrapper
from src.models.neural.tf_wrapper import DRATTNWrapper

from src.models.linear.plain_regression import RegularizedRegression
from src.models.linear.fixed_regression import FixedRegression
from src.models.linear.confound_regression import ConfoundRegression
from src.models.linear.double_regression import DoubleRegression


MODEL_CLASSES = {
    'A_ATTN': AATTNWrapper,
    'A_BOW': ABOWWrapper,
    'DR_ATTN': DRATTNWrapper,
    'DR_BOW': DRBOWWrapper,
    'fixed-regression': FixedRegression,
    'confound-regression': ConfoundRegression,
    'regression': RegularizedRegression,
    'double-regression': DoubleRegression,
}

