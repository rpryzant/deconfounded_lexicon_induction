"""Code for creating an manipulating tensorflow graphs and sessions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import os
import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm

import sys; sys.path.append('../..')

from src.models.abstract_model import Model
from src.models.abstract_model import Prediction
import src.models.neural.a_attn as A_ATTN
import src.models.neural.a_bow as A_BOW
import src.models.neural.a_cnn as A_CNN
import src.models.neural.dr_attn as DR_ATTN
import src.models.neural.dr_bow as DR_BOW

import src.msc.utils as utils

# Disable TensorFlow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TFModelWrapper(Model):
  """A wrapper object for TF models that manipulates graphs and sessions."""

  def __init__(self, config, params):
    self.config = config
    self.params = params

  def save(self, model_dir):
    """Saves a model checkpoint into `model_dir`."""
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    self.loaded_model.model.saver.save(
        self.sess,
        os.path.join(model_dir, 'model.ckpt'),
        global_step=self.global_step)
    print('TF WRAPPER: saved into %s' % model_dir)

  def load(self, dataset, model_dir):
    """Loads a model checkpoint from `model_dir`."""
    (self.loaded_model, self.global_step,
     self.sess) = self.create_or_load_model(model_dir, dataset)
    return self.loaded_model

  def create_or_load_model(self, model_dir, dataset):
    """Tries to load a model, and if that fails, initializes a new one."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)

    model = self.model_builder_class.build_model_graph(self.config, self.params,
                                                       dataset)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=model.graph, config=config)

    with model.graph.as_default():
      tf.set_random_seed(self.config.seed)

      if latest_ckpt:
        start_time = time.time()
        model.model.saver.restore(sess, latest_ckpt)
        sess.run(tf.tables_initializer())
        print('INFO: loaded model parameters from %s, time %.2fs' % (
                     latest_ckpt,
                     time.time() - start_time))
      else:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print('INFO: created model with fresh parameters, time %.2fs' % (
                     time.time() - start_time))

    print('INFO: trainable variables:')
    values = sess.run(model.model.trainable_variable_names)
    for name, value in zip(model.model.trainable_variable_names, values):
      print('\t%s   ----    Shape: %s' % (name, value.shape))

    global_step = model.model.global_step.eval(session=sess)
    return model, global_step, sess

  def train(self, dataset, model_dir):
    """Trains the model on a dataset."""
    (self.loaded_model, self.global_step,
     self.sess) = self.create_or_load_model(model_dir, dataset)
    summary_writer = tf.summary.FileWriter(
        os.path.join(model_dir, 'train_log'), self.loaded_model.graph)

    self.sess.run(self.loaded_model.model.iter['initializer'])

    epochs = 0
    start_time = time.time()
    progbar = tqdm(total=self.params['num_train_steps'])
    while self.global_step < self.params['num_train_steps']:
      try:
        self.global_step, _, summary = self.loaded_model.model.train(self.sess)
        progbar.update(1)
        summary_writer.add_summary(summary, self.global_step)
      except tf.errors.OutOfRangeError:
        epochs += 1
        print('epoch %s took %.2fs' % (epochs, time.time() - start_time))
        start_time = time.time()
        self.sess.run(self.loaded_model.model.iter['initializer'])
    progbar.close()

  def inference(self, dataset, model_dir, dev=True):
    """Uses the model to perform inference over a dataset."""
    # TODO(rpryzant) -- refactor?
    self.sess.run(self.loaded_model.model.iter['initializer'])

    predictions = defaultdict(list)
    tok_importance = defaultdict(dict)

    try:
      while True:
        (batch_predictions,
         batch_tok_importance) = self.loaded_model.model.inference_on_batch(
             self.sess)
        for variable_name in batch_predictions:
          # Remember the predictions on this batch.
          predictions[variable_name] += batch_predictions[variable_name]

          # Remember the scores on each token as well.
          if dataset.get_variable(variable_name)['type'] == utils.CATEGORICAL:
            for level_name in dataset.class_to_id_map[variable_name]:

              # This means there weren't any true positives for this level
              # in the batch so we didn't save any attention scores into
              # the corresponding dictionary.
              if level_name not in batch_tok_importance[variable_name]:
                continue

              if level_name not in tok_importance[variable_name]:
                tok_importance[variable_name][
                    level_name] = batch_tok_importance[variable_name][
                        level_name]
              else:
                for token, scores in batch_tok_importance[variable_name][
                    level_name].items():
                  if token not in tok_importance[variable_name][level_name]:
                    tok_importance[variable_name][level_name][token] = scores
                  else:
                    tok_importance[variable_name][level_name][token] += scores
          else:
            for token, scores in batch_tok_importance[variable_name].items():
              if token not in tok_importance[variable_name]:
                tok_importance[variable_name][token] = scores
              else:
                tok_importance[variable_name][token] += scores

    except tf.errors.OutOfRangeError:
      pass

    # Aggregate the feature importance scores.
    for variable_name in tok_importance:
      if dataset.get_variable(variable_name)['type'] == utils.CATEGORICAL:
        for level in tok_importance[variable_name]:
          for token in tok_importance[variable_name][level]:
            tok_importance[variable_name][level][token] = np.mean(
                tok_importance[variable_name][level][token])
      else:
        for token in tok_importance[variable_name]:
          tok_importance[variable_name][token] = tok_importance[variable_name][
              token]

    return Prediction(scores=predictions, feature_importance=tok_importance)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Model-specific wrapper subclasses which basically just point
# the model building function to a specific graph building class.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ABOWWrapper(TFModelWrapper):

  def __init__(self, config, params):
    TFModelWrapper.__init__(self, config, params)

    self.model_builder_class = A_BOW.ABow


class AATTNWrapper(TFModelWrapper):

  def __init__(self, config, params):
    TFModelWrapper.__init__(self, config, params)

    self.model_builder_class = A_ATTN.AAttn


class ACNNWrapper(TFModelWrapper):

  def __init__(self, config, params):
    TFModelWrapper.__init__(self, config, params)

    self.model_builder_class = A_CNN.ACnn


class DRBOWWrapper(TFModelWrapper):

  def __init__(self, config, params):
    TFModelWrapper.__init__(self, config, params)

    self.model_builder_class = DR_BOW.DrBow


class DRATTNWrapper(TFModelWrapper):

  def __init__(self, config, params):
    TFModelWrapper.__init__(self, config, params)

    self.model_builder_class = DR_ATTN.DrAttn
