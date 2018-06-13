"""Code for pulling words out of trained models and evaluating them."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
import sklearn.metrics
import sys; sys.path.append('../..')
import src.analysis.stats as stats


def run_model(config, dataset, features, eval_variable_name, eval_level_name,
              confounds):
  """Trains and tests a regression model.

  TODO -- assumes binary outcomes!!

  Args:
    config: A config.yaml file which has been parsed into an object.
    dataset: A data.dataset.Dataset object.
    features: A list of tokens to be used as features in the model.
    eval_variable_name: string, the name of the variable we are predicting.
    eval_level_name: string, the level of the variable we are predicting.
    confounds: list(dict), members of config.data_spec which are confounds.

  Returns:
    The cross-validated loss of three models on the data. Models are trained
      with (1) confound features, (2) text + confounds, and (3) text only.
  """
  m = linear_model.LogisticRegression()
  dataset.set_active_split(config.train_suffix)
  kfold = model_selection.KFold(n_splits=10, random_state=7)

  def scorer(estimator, x, y):
    yhats = estimator.predict_proba(x)
    return sklearn.metrics.log_loss(y, yhats, labels=[0, 1])

  eval_level_index = dataset.class_to_id_map[eval_variable_name][
      eval_level_name]
  y = np.squeeze(dataset.np_data[config.train_suffix][eval_variable_name]
                 [:, eval_level_index].toarray())

  retain_indices = [dataset.features[f] for f in features]
  x_text = dataset.np_data[config.train_suffix]['text-input'].toarray()
  x_text = x_text[:, retain_indices]

  x_confound = None
  for variable in confounds:
    variable_data = dataset.np_data[config.train_suffix][variable[
        'name']].toarray()
    if x_confound is None:
      x_confound = variable_data
    else:
      x_confound = np.column_stack([x_confound, variable_data])

  x_both = np.column_stack([x_text, x_confound])

  print('EVALUATOR: running linear model with selected features.')
  text_xentropy = model_selection.cross_val_score(
      m, x_text, y, cv=kfold, scoring=scorer).mean()

  print('EVALUATOR: running confound model.')
  confound_xentropy = model_selection.cross_val_score(
      m, x_confound, y, cv=kfold, scoring=scorer).mean()

  print('EVALUATOR: running fixed model with selected features.')
  both_xentropy = model_selection.cross_val_score(
      m, x_both, y, cv=kfold, scoring=scorer).mean()

  return text_xentropy, confound_xentropy, both_xentropy


def evaluate(config,
             dataset,
             predictions,
             model_dir,
             eval_variable_name,
             eval_level_name=''):
  """Evaluates the predictions of a trained model.

  Evaluation consists of the following:
    (1) Harvest the best (and worst) words from the predictions.
    (2) Train regressions with (a) just these words, (b) these words and
        counfounds, and (c) just confounds.
    (3) Get the correlation between these words and each outcome variable
        we are "controlling for".
    (4) Get the performance of models from step (2) on each outcome variable
        we are *not* controlling for.

  Args:
    config: A config.yaml file which has been parsed into an object.
    dataset: A data.dataset.Dataset object.
    predictions: An instance of models.abstract_model.Prediction,
                 which holds per-example predictions, as well as
                 an "importance value" for each feature.
    model_dir: string, A path to the model's working directory.
    eval_variable_name: string, the name of the variable we are evaluating.
    eval_level_name: string, the name of the categorical level we are evaluating
      "against". This is optional. If not provided then the system will assume
      that `eval_variable_name` corresponds to a continuous variable.

  Returns:
    results: A dictionary that maps metric names to their values.
  """
  print('EVALUATOR: evaluating words for %s, level %s' % (
               eval_variable_name, eval_level_name))
  # Point the dataset at the test data.
  pre_eval_split = dataset.current_split
  dataset.set_active_split(config.test_suffix)

  # Select the K most and least important features.
  # Note that K = config.num_eval_features.
  all_features = sorted(
      predictions.feature_importance.items(), key=lambda x: x[1])[::-1]
  top_features = all_features[:config.num_eval_features]

  # Filter out features which aren't in the vocabulary (contained in
  # dataset.features). This ensures that features like intercepts and
  # confound levels are ignored.
  top_features = [x[0] for x in top_features if x[0] in dataset.features]

  # Write the feature list.
  with open(
      os.path.join(
          model_dir,
          '%s|%s_top_words.txt' % (eval_variable_name, eval_level_name)),
      'w') as f:
    f.write('\n'.join([
        '%s\t%.4f' % (f, predictions.feature_importance[f])
        for f in top_features
    ]))

  # Get the average log-odds between the selected features and each
  # level of the outcome variable.
  outcome = next(
      (variable for variable in config.data_spec[1:]
       if variable['name'] == eval_variable_name and eval_level_name))
  confounds = [
      variable for variable in config.data_spec[1:]
      if not variable['skip'] and variable['control']
  ]

  # Evaluate the selected feature's ability to predict the outcome.
  text_xentropy, confound_xentropy, both_xentropy = run_model(
      config, dataset, top_features, eval_variable_name, eval_level_name,
      confounds)

  # Get the log-odds between each word in `top_features` and
  # the target outcome (level `eval_level_name` of variable
  # `eval_variable_name`). Write these values to a file and also take
  # their average.
  target_log_odds = []
  with open(
      os.path.join(
          model_dir,
          '%s|%s_log_odds.txt' % (eval_variable_name, eval_level_name)),
      'a') as debug_file:
    for f in top_features:
      result = stats.log_odds(f, outcome['name'], dataset, config)
      debug_file.write('%s\t%s\n' % (f, str(result)))
      target_log_odds.append(result[eval_level_name])

  mean_target_log_odds = np.mean(target_log_odds)

  dataset.set_active_split(pre_eval_split)

  return {
      'mu_target_log_odds': mean_target_log_odds,
      'confound_correlations': -1,
      'target_correlatoins': -1,
      'mu_confound_corr': -1,
      'mu_target_corr': -1,
      'performance': -1,
      'mu_reg_perf': text_xentropy,
      'mu_fixed_perf': both_xentropy,
      'mu_confound_perf': confound_xentropy
  }
