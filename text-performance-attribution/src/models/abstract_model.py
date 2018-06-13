"""Abstract interface that all models must implement."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

# Prediction is a little object to represent model inferences.
# Fields:
#   scores: dict(string => np array or list)
#     response variable name => [prediction per example on that response
#            variable]
#
#   feature_importance: dict(string => dict(string => list))
#     variable name => feature name => importance value for that feature
#     i.e., what are the importance values for each feature with respect to
#       each outcome variable? Note that in the case of categorical outcome
#       variables, this will be a triple nested dictionary which maps
#       variable name => level => feature name => score.
Prediction = namedtuple('Prediction', ('scores', 'feature_importance'))


class Model(object):
  """Superclass for all models."""

  def __init__(self, config, params):
    self.config = config
    self.params = params

  def save(self, directory):
    """Saves a representation of the model into a directory."""
    raise NotImplementedError

  def load(self, dataset, model_dir):
    """Restores a representation of the model from a directory.

    Args:
      dataset: src.data.dataset.Dataset
      model_dir: string, a directory that contains a model checkpoint.
    """
    raise NotImplementedError

  def train(self, dataset, model_dir):
    """Trains the model using the data in `dataset`.

    Instead of returning something, this method will update a private
    representation of its parameters.

    Args:
      dataset: src.data.dataset.Dataset
      model_dir: string, directory where the system should write logs,
        checkpoints, etc.
    """
    raise NotImplementedError

  def inference(self, dataset, model_dir):
    """Runs inference on whichever split the dataset is configured for.

    Args:
      dataset: src.data.dataset.Dataset
      model_dir: string, a directory where the system should write logs,
        checkpoints, etc.

    Returns:
      An abstract_model.Prediction namedtuple.
    """
    raise NotImplementedError
