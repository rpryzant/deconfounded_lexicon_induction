"""This is the main binary to start training and testing for attribution."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import copy
import csv
import os
import random
import sys
import time
from absl import flags
import numpy as np

import tensorflow as tf

import src.analysis.evaluator as evaluator
from src.data.dataset import Dataset
from src.models.abstract_model import Prediction

from src.models.linear.confound_regression import ConfoundRegression
from src.models.linear.double_regression import DoubleRegression
from src.models.linear.fixed_regression import FixedRegression
from src.models.linear.regression_base import Regression

from src.models.neural.tf_wrapper import AATTNWrapper
from src.models.neural.tf_wrapper import ABOWWrapper
from src.models.neural.tf_wrapper import ACNNWrapper
from src.models.neural.tf_wrapper import DRATTNWrapper
from src.models.neural.tf_wrapper import DRBOWWrapper

import src.msc.utils as utils

# These are the mappings from model names as specified in a config.yaml file
# to the appropriate class for this model.
_MODEL_CLASSES = {
    'A_ATTN': AATTNWrapper,
    'A_BOW': ABOWWrapper,
    'A_CNN': ACNNWrapper,
    'DR_ATTN': DRATTNWrapper,
    'DR_BOW': DRBOWWrapper,
    'fixed-regression': FixedRegression,
    'confound-regression': ConfoundRegression,
    'regression': Regression,
    'double-regression': DoubleRegression,
}
# How many times we are allowed to expand a model template
# for hyperparameter sweeps.
_MODEL_EXPANSION_LIMIT = 30

FLAGS = flags.FLAGS
flags.DEFINE_string('yaml_config', 'config.yaml',
                    'Config file for this experiment.')
flags.DEFINE_bool('train', True, 'Run training pipeline.')
flags.DEFINE_bool('test', True, 'Run testing pipeline.')
FLAGS(sys.argv)




def set_seed(seed):
  """Sets seeds for all of the random number generators."""
  random.seed(seed)
  np.random.seed(seed)
  # NOTE: this only seeds the default graph.
  tf.set_random_seed(seed)


def train_model(model_description, config, dataset):
  """Initializes, trains, and saves a model.

  The parameters of the model are saved into a subdirectory of the working dir.

  Args:
    model_description: dict, a dictionary corresponding to the part of a
      config which is about the current model which is to be run.
      Note that this dictionary is a subset of the `config`
      argument, but we keep it separate here for readability.
    config: NamedTuple, a config.yaml file which has been parsed into an object.
    dataset: src.data.dataset.Dataset, a convenience class for managing data.
  """
  print('MAIN: training %s' % model_description['name'])
  # Point the dataset at the "train" data.
  dataset.set_active_split(config.train_suffix)

  # Create an output dir for this model.
  start_time = time.time()
  model_dir = os.path.join(config.working_dir, model_description['name'])
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  # Initialize, train, and save.
  model = _MODEL_CLASSES[model_description['type']](
      config=config, params=model_description)
  model.train(dataset, model_dir)
  model.save(model_dir)
  print('MAIN: training %s done, time %.2fs'%( model_description['name'],
               time.time() - start_time))


def test_model(model_description, config, dataset):
  """Loads a model, do inference with it, and compute evaluation metrics.

  Args:
    model_description: dict, A dictionary corresponding to the part of a config
      which is about the current model which is to be run.
      Note that this dictionary is a subset of the `config`
      argument, but we keep it separate here for readability.
    config: NamedTuple, a config.yaml file which has been parsed into an object.
    dataset: src.data.dataset.Dataset, a convenience class for managing data.

  Returns:
    result: dict, a dictionary containing metrics and information about the run.
  """
  def make_summary_dict(evaluation, target_variable):
    """Finalize an evaluation before it is written to the summary file."""
    return {
        'target_variable':
            target_variable,
        'model_name':
            model_description['name'],
        'model_type':
            model_description['type'],
        'params':
            str(model_description),
        'target_log_odds':
            evaluation['mu_target_log_odds'],
        'confound_correlation':
            evaluation['mu_confound_corr'],
        'target_correlation':
            evaluation['mu_target_corr'],
        'informativeness_coef':
            evaluation['mu_confound_perf'] - evaluation['mu_fixed_perf'],
        'regression_performance':
            evaluation['mu_reg_perf'],
        'fixed_performance':
            evaluation['mu_fixed_perf'],
        'confound_performance':
            evaluation['mu_confound_perf'],
        'model_dir':
            model_dir,
    }

  print('MAIN: inference with %s' % model_description['name'])

  start_time = time.time()

  # Point the dataset towards its proper "test" data.
  if 'CNN' in model_description['type']  or 'ATTN' in model_description['type']:
    # Attention and CNN models perform inference over the training data.
    dataset.set_active_split(config.train_suffix)
  else:
    # All other models perform inference over the test set.
    dataset.set_active_split(config.test_suffix)

  # Load the model.
  model = _MODEL_CLASSES[model_description['type']](
      config=config, params=model_description)
  model_dir = os.path.join(config.working_dir, model_description['name'])
  model.load(dataset, model_dir)

  # Run inference with the loaded model.
  predictions = model.inference(dataset, model_dir)
  utils.pickle(predictions, os.path.join(model_dir, 'predictions'))

  # Evaluate these predictions and write the results.
  evaluation_reports = []
  eval_variable = dataset.get_variable(config.eval_variable_name)

  # Make sure the user asked us to evaluate a target variable.
  assert eval_variable['control'] is False

  if eval_variable['type'] == utils.CONTINUOUS:
    # If the eval variable is continuous, then create a new Prediction
    # object with (1) the original predictions from the trained model, and
    # (2) the feature importance values for the eval variable.
    eval_predictions = Prediction(
        scores=predictions.scores,
        feature_importance=predictions.feature_importance[
            config.eval_variable_name])
    # Now that we have a Prediction object whose feature_importance field
    # is a {token => score} mapping for the eval_variable we run an evaluation.
    evaluation = evaluator.evaluate(config, dataset, eval_predictions,
                                    model_dir, config.eval_variable_name)
    with open(
        os.path.join(model_dir, '%s_summary.txt' % (config.eval_variable_name)),
        'w') as f:
      f.write(str(evaluation))
    # Add the eval for this variable to the outgoing report.
    evaluation_reports.append(
        make_summary_dict(evaluation, config.eval_variable_name))

  # This is the branch we take if the eval variable is categorical.
  else:
    # Perform an evaluation for each level of this categorical variable
    # because the model may have generated different feature_importance
    # values for each leve.
    for level in predictions.feature_importance[config.eval_variable_name]:
      variable_full_name = '%s|%s' % (config.eval_variable_name, level)

      # Create a "flat" Prediction object whose feature_importance field
      # is a {token => score} mapping and the scores are specific to this
      # variable and level.
      eval_predictions = Prediction(
          scores=predictions.scores,
          feature_importance=predictions.feature_importance[
              config.eval_variable_name][level])
      # Run an evaluation with this variable/level specific Predictions object.
      evaluation = evaluator.evaluate(
          config,
          dataset,
          eval_predictions,
          model_dir,
          config.eval_variable_name,
          eval_level_name=level)
      with open(
          os.path.join(model_dir, '%s_summary.txt' % (variable_full_name)),
          'w') as f:
        f.write(str(evaluation))
      # Add the eval for this variable and level to the outgoing report.
      evaluation_reports.append(
          make_summary_dict(evaluation, variable_full_name))

  print('MAIN: evaluation %s done, time %.2fs' % (
               model_description['name'],
               time.time() - start_time))

  return evaluation_reports


def run_experiment(config, run_training, run_testing):
  """Runs an experiment specified by a config file and cli args.

  Args:
    config: NamedTuple, a config.yaml file which has been parsed into an object.
    run_training: bool, whether to run the training pipeline.
    run_testing: bool, whether to run the testing pipeline.

  Returns:
    results: list(dict), a list of dictionaries, one per model specified in
      the `config`. Each dictionary contains performance metrics and
      hyperparameters which are specific to that model.
  """
  # Boilerplate: set seeds and create working dir.
  set_seed(config.seed)
  if not os.path.exists(config.working_dir):
    os.MakeDirs(config.working_dir)
  utils.write_config(config, os.path.join(config.working_dir, 'config.yaml'))

  print('MAIN: parsing dataset')
  start = time.time()
  dataset = Dataset(config, config.working_dir)
  print('MAIN: dataset done. took %.2fs' % (time.time() - start))

  # Train & test each of the models which are listed in the config.
  results = []
  for model_description in config.model_spec:
    if model_description.get('skip', False):
      continue
   # try:

    if run_training:
      train_model(model_description, config, dataset)
    if run_testing:
      results += test_model(model_description, config, dataset)

    #except Exception as e:
    #  print(str(e))

  return results


def validate_tsv_data_file(config, in_path, out_path):
  """Combs through a TSV data file and writes its valid rows to `out_path`.

  This method won't do anything if there's already a file at `out_path`.

  Args:
    config: NamedTuple, a config.yaml file which has been parsed into an object.
    in_path: string, a file path pointing to a TSV which is described by
      config.data_spec.
    out_path: string, a file path where the valid rows of `in_path` are to
      be written.
  Returns:
    Nothing.
  """
  if os.path.exists(out_path):
    return

  text_inputs = set()

  with open(out_path, 'w') as out_file:
    out_file = open(out_path, 'w')

    for line in open(in_path):
      skip = False
      parts = line.strip().split('\t')

      # Check for invalid number of columns.
      if len(config.data_spec) != len(parts):
        skip = True

      # Ignore repeated rows.
      text_input = parts[0]
      if text_input in text_inputs:
        skip = True
      else:
        text_inputs.add(text_input)

      # Loop through each column and make sure it is valid.
      for x, column in zip(parts, config.data_spec):
        # Ignore skipped columns.
        if column.get('skip', False):
          continue

        # Ignore rows with empty values or non-numeric continuous values.
        if not x or (column['type'] == utils.CONTINUOUS and
                     not utils.is_number(x)):
          skip = True

      if skip is False:
        # Normalize (lowercase).
        line = '\t'.join([parts[0].lower()] + parts[1:]) + '\n'

        out_file.write(line)


def prepare_data_and_update_config(config):
  """Cleans and splits the data, then points the config to the new data.

  Args:
    config: NamedTuple, a config.yaml file which has been parsed into an object.

  Returns:
    config: a new version of the passed `config` object with a new field
      (data_dir). This field points to a subdirectory of the working directory
      which contains all of the data for the experiment to be run.
  """
  out_data_dir = os.path.join(config.working_dir, 'data')
  in_data_path = config.data_path
  data_file_name = os.path.basename(in_data_path)
  out_path_base = os.path.join(out_data_dir, data_file_name + '.validated')

  if not os.path.exists(out_data_dir):
    os.makedirs(out_data_dir)

  print('MAIN: making splits...')
  validate_tsv_data_file(config, in_data_path, out_path_base)

  utils.split_in_three(out_path_base, config.dev_size, config.test_size,
                       out_path_base + config.train_suffix,
                       out_path_base + config.dev_suffix,
                       out_path_base + config.test_suffix)

  # Update the config by pointing it at the new data dir and also
  #   telling it to look for validated files.
  config_dict = copy.deepcopy(dict(config._asdict()))
  config_dict['data_path'] = out_path_base
  config = namedtuple('config', config_dict.keys())(**config_dict)

  return config


def expand_model_parameters(model_params, expansion_limit):
  """Expand a model template into random combinations of hyperparameters.

  For example, if model_params is:
    {name: regression, l2_strength: [0, 1], train_steps: [100, 50] }

  Then the returned new_model_params will be the following:
  [
    {name: regression_0, l2_strength: 0, train_steps: 50 },
    {name: regression_1, l2_strength: 0, train_steps: 100 }
    {name: regression_2, l2_strength: 1, train_steps: 50 }
    {name: regression_3, l2_strength: 1, train_steps: 100 }
  ]

  Args:
    model_params: dict(string => list/string/int/float), a member of
      config.model_spec and the template we want to expand.
    expansion_limit: int, the number of time we are allowed to expand this
      model.

  Returns:
    new_model_params: list(dict), up to `expansion_limit` instantations
      of the provided model parameters.
  """
  new_model_params = []

  def random_spec(kv_pairs, model_id):
    out = []
    for key, value in kv_pairs:
      if key == 'name':
        value += '_%s' % model_id
      if isinstance(value, list):
        out += [(key, random.choice(value))]
      else:
        out += [(key, value)]
    return out

  n_config_options = sum(
      len(x) if isinstance(x, list) else 0 for x in model_params.values()) + 1
  for i in range(min(expansion_limit, n_config_options)):
    new_model_params += [dict(random_spec(model_params.items(), i))]

  return new_model_params


def expand_model_spec(parent_config):
  """Expands a yaml config to include all possible hyperparameter combinations.

  The `model_spec` key of a yaml config contains a list of dictionaries (one
  per model) that maps hyperparameters to their values. If the user gives a list
  for any of these values, than it is assumed that the users wants to do a
  hyperparameter search over various combinations therein. This function
  expands each dictionary into the possible hyperparameter combinations it
  describes.

  For example, if config.model_spec contains:
  [
    {name: regression, l2_strength: [0, 1], train_steps: [100, 50] }
  ]

  Then the returned config.model_spec will have the following:
  [
    {name: regression_0, l2_strength: 0, train_steps: 50 },
    {name: regression_1, l2_strength: 0, train_steps: 100 }
    {name: regression_2, l2_strength: 1, train_steps: 50 }
    {name: regression_3, l2_strength: 1, train_steps: 100 }
  ]

  This function is not exaustive and will cut off the expansion at
  _MODEL_EXPANSION_LIMIT models per provided template.

  Args:
    parent_config: NamedTuple, a config.yaml file which has been parsed
      into an object.

  Returns:
    new_config: NamedTuple, a new config.yaml file with expanded model_spec
      fields.
  """
  config_dict = copy.deepcopy(dict(parent_config._asdict()))
  new_model_spec = []
  model_spec = config_dict['model_spec']

  for model_parameters in model_spec:
    new_model_spec += expand_model_parameters(model_parameters,
                                              _MODEL_EXPANSION_LIMIT)
  random.shuffle(new_model_spec)
  config_dict['model_spec'] = new_model_spec
  new_config = namedtuple('config', config_dict.keys())(**config_dict)
  return new_config


def main():
  # Parse the config.
  config = utils.load_yaml_config(FLAGS.yaml_config)

  # Boilerplate.
  reload(sys)
  sys.setdefaultencoding('utf8')

  if not os.path.exists(config.working_dir):
    os.makedirs(config.working_dir)

  # Validate and split the data. This will create a new copy of the data, with
  # the data for each variable in a different file. The config's `data_dir`
  # will also be modified to point to these new data files.
  print('MAIN: validating data...')
  start = time.time()
  config = prepare_data_and_update_config(config)
  print('MAIN: validation done. Took %.2fs.' % (time.time() - start))

  # Set the system up for a hyperparameter search if the user asked for it.
  print('MAIN: expanding model spec...')
  start = time.time()
  config = expand_model_spec(config)
  print('MAIN: expansion done. Took %.2fs.' % (time.time() - start))

  # Boilerplate.
  start = time.time()
  summary_path = os.path.join(config.working_dir, 'summary.csv')
  summary_file = open(summary_path, 'a')
  csv_writer = csv.writer(summary_file)

  # Run the experiment and accumulate its results.
  results = run_experiment(config, FLAGS.train, FLAGS.test)

  if FLAGS.test:
    print('MAIN: writing summary to %s', summary_path)

    # Write results in sorted order to guarentee columns won't be mixed up.
    csv_writer.writerow(sorted(results[0].keys()))
    for result_dict in results:
      csv_writer.writerow(
          [result_dict[key] for key in sorted(result_dict.keys())])

  summary_file.close()


if __name__ == '__main__':
  main()
