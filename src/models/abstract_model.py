from collections import namedtuple


""" little object to represent model inference

    scores: 
        {
            response variable name: [prediction per example on that response variable]
        }
    feature_importance:
        {
            feature name: importance value for that feature
        }
"""
Prediction = namedtuple(
    'Prediction',
    ('scores', 'feature_importance'))
 

class Model(object):
    """ superclass for all models 
    """
    def __init__(self, config, params):
        self.config = config
        self.params = params


    def save(self, dir):
        """ saves a representation of the model into a directory
        """
        raise NotImplementedError


    def load(self, dataset, model_dir):
        """ restores a representation of the model from dir
        """
        raise NotImplementedError


    def train(self, dataset, model_dir):
        """ trains the model using a src.data.dataset.Dataset
            saves model-specific metrics (loss, etc) into self.report

            returns nothing, but updates some kind of inner self.model
        """
        raise NotImplementedError


    def inference(self, dataset, model_dir):
        """ run inference on whichever split the dataset is configured for

            returns a abstract_model.Prediction
        """
        raise NotImplementedError

