"""Tests for src.data.dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import os
import random
from absl.testing import absltest
import numpy as np
import tensorflow as tf
import pytest
import unittest

import src; src.path.append('../..')
from src.data import dataset


def create_test_data(working_dir, bigrams=False):
  """Create mock data for testing, writing it to `working_dir`."""
  if not os.path.exists(working_dir):
    os.makedirs(working_dir)

  if bigrams:
    data = [
        ['is_not not_be be_you', '1.0', '3.1', 'a', '_1'],  # train example.
        ['is_not not_be be_you', '1.1', '3.2', 'a', '_1'],  # train example.
        ['is_not not_an an_sun', '1.2', '2.3', 'b', '_2'],  # dev example.
        ['is_fox fox_if if_see', '1.3', '2.4', 'a', '_3']  # test example.
    ]
  else:
    # `is` has frequency 4, `not` has frequency 3, all others occur at most 2x.
    data = [
        ['is not be you', '1.0', '3.1', 'a', '_1'],  # train example.
        ['is not be you', '1.1', '3.2', 'a', '_1'],  # train example.
        ['is not an sun', '1.2', '2.3', 'b', '_2'],  # dev example.
        ['is fox if see', '1.3', '2.4', 'a', '_3']  # test example.
    ]

  with open(working_dir + '/data', 'w') as f:
    f.write('\n'.join(['\t'.join(row) for row in data]))

  with open(working_dir + '/data.train', 'w') as f:
    f.write('\n'.join(['\t'.join(row) for row in data[:2]]) + '\n')

  with open(working_dir + '/data.dev', 'w') as f:
    f.write('\t'.join(data[2]) + '\n')

  with open(working_dir + '/data.test', 'w') as f:
    f.write('\t'.join(data[3]) + '\n')

  return data, data[:2], data[2], data[3]


def create_test_config(working_dir):
  """Create a mock config for testing."""
  # See //learning/aimee/marilyn/creative_text_attribution/sample_config.yaml
  # for config documentation.
  config_mock_dict = {
      'data_spec': [{
          'type': 'text',
          'name': 'input'
      }, {
          'type': 'continuous',
          'control': False,
          'skip': False,
          'name': 'continuous_target'
      }, {
          'type': 'continuous',
          'control': True,
          'skip': False,
          'name': 'continuous_confound'
      }, {
          'type': 'categorical',
          'control': False,
          'skip': False,
          'name': 'categorical_target'
      }, {
          'type': 'categorical',
          'control': True,
          'skip': False,
          'name': 'categorical_confound'
      }],
      'vocab': {
          'vocab_file': None,
          'top_n': 2
      },
      'max_seq_len':
          3,
      'unk':
          '<unk>',
      'eos':
          '</s>',
      'working_dir':
          working_dir,
      'train_suffix':
          '.train',
      'dev_suffix':
          '.dev',
      'test_suffix':
          '.test',
      'data_path':
          os.path.join(working_dir, 'data'),
  }
  return namedtuple('config', config_mock_dict.keys())(**config_mock_dict)


def make_dataset_and_config(directory, bigrams=False):
  """Create skeleton dataset and config objects for testing."""
  # Mock data for testing.
  create_test_data(directory, bigrams=bigrams)

  # Mock config containing the minimum fields needed to instantiate a
  # Dataset object.
  config_mock = create_test_config(directory)

  # Initialize a Dataset object for testing.
  dataset_mock = dataset.Dataset(config_mock, directory)

  return dataset_mock, config_mock


class DatasetTest(unittest.TestCase):

  def setUp(self):
    random.seed(1)

    self.tmp_dir = absltest.get_default_test_tmpdir()
    self.dataset, self.config_mock = make_dataset_and_config(self.tmp_dir)

  def test_dataset_file_pointers(self):
    """Ensures the dataset creates separate files for each column of data."""

    # Make sure the datset's split sizes are correct.
    for split, size in zip(['.train', '.dev', '.test'], [2, 1, 1]):
      self.assertEqual(self.dataset.split_sizes[split], size)

    # There should be files for each non-input, non-confound variable.
    for split_name in ['.train', '.dev', '.test']:
      for variable in self.config_mock.data_spec[1:]:
        if variable['control']:
          continue

        var_name = variable['name']
        data_path = self.config_mock.data_path + '.' + var_name + split_name
        self.assertEqual(self.dataset.data_files[split_name][var_name],
                         data_path)

    # There should be files for every variable containing *all* of the data
    # (not train/test/dev splits).
    for variable in self.config_mock.data_spec:
      variable_name = variable['name']
      data_path = self.config_mock.data_path + '.' + variable_name
      self.assertEqual(self.dataset.whole_data_files[variable_name], data_path)

  def test_dataset_categorical_variable_ids(self):
    """Ensures the dataset assigned IDs to categorical variables correctly."""

    # The levels of each categorical variable should have been assigned IDs
    # in alphanumeric order.
    c2id = self.dataset.class_to_id_map

    self.assertEqual(c2id['categorical_confound']['_1'], 0)
    self.assertEqual(c2id['categorical_confound']['_2'], 1)
    self.assertEqual(c2id['categorical_confound']['_3'], 2)

    self.assertEqual(c2id['categorical_target']['a'], 0)
    self.assertEqual(c2id['categorical_target']['b'], 1)

    id2c = self.dataset.id_to_class_map

    self.assertEqual(id2c['categorical_confound'][0], '_1')
    self.assertEqual(id2c['categorical_confound'][1], '_2')
    self.assertEqual(id2c['categorical_confound'][2], '_3')

    self.assertEqual(id2c['categorical_target'][0], 'a')
    self.assertEqual(id2c['categorical_target'][1], 'b')

  def test_dataset_categorical_variable_parse(self):
    """Ensures the dataset parsed categorical data correctly."""
    # If example i belongs to class id j, then dataset.np_data[i, j] == 1.

    self.assertTrue(
        np.array_equal(
            self.dataset.np_data['.train']['categorical_confound'].toarray(),
            np.array([[1, 0, 0], [1, 0, 0]])))
    self.assertTrue(
        np.array_equal(
            self.dataset.np_data['.dev']['categorical_confound'].toarray(),
            np.array([[0, 1, 0]])))
    self.assertTrue(
        np.array_equal(
            self.dataset.np_data['.test']['categorical_confound'].toarray(),
            np.array([[0, 0, 1]])))

    self.assertTrue(
        np.array_equal(
            self.dataset.np_data['.train']['categorical_target'].toarray(),
            np.array([[1, 0], [1, 0]])))
    self.assertTrue(
        np.array_equal(
            self.dataset.np_data['.dev']['categorical_target'].toarray(),
            np.array([[0, 1]])))
    self.assertTrue(
        np.array_equal(
            self.dataset.np_data['.test']['categorical_target'].toarray(),
            np.array([[1, 0]])))

  def test_dataset_continuous_variable_parse(self):
    """Ensures the dataset parsed continuous data correctly."""
    self.assertTrue(
        np.array_equal(
            np.squeeze(self.dataset.np_data['.train']['continuous_confound']
                       .toarray()), np.array([3.1, 3.2])))
    self.assertEqual(
        np.squeeze(
            self.dataset.np_data['.dev']['continuous_confound'].toarray()), 2.3)
    self.assertEqual(
        np.squeeze(
            self.dataset.np_data['.test']['continuous_confound'].toarray()),
        2.4)

    self.assertTrue(
        np.array_equal(
            np.squeeze(
                self.dataset.np_data['.train']['continuous_target'].toarray()),
            np.array([1.0, 1.1])))
    self.assertEqual(
        np.squeeze(self.dataset.np_data['.dev']['continuous_target'].toarray()),
        1.2)
    self.assertEqual(
        np.squeeze(
            self.dataset.np_data['.test']['continuous_target'].toarray()), 1.3)

  def test_dataset_vocab_generation(self):
    """Ensures the dataset generated the proper vocabulary."""
    # Vocab is of proper size (+1 for unk token).
    self.assertEqual(self.dataset.vocab_size,
                     self.config_mock.vocab['top_n'] + 1)
    self.assertEqual(len(self.dataset.features), self.dataset.vocab_size)

    # dataset.features is a map from token to id. The ids of these
    # vocab items should correspond to their frequency in the data.
    self.assertEqual(self.dataset.features['<unk>'], 0)
    self.assertEqual(self.dataset.features['is'], 1)
    self.assertEqual(self.dataset.features['not'], 2)

    # dataset.ids_to_features is a map from id to token.
    self.assertEqual(self.dataset.ids_to_features[0], '<unk>')
    self.assertEqual(self.dataset.ids_to_features[1], 'is')
    self.assertEqual(self.dataset.ids_to_features[2], 'not')

    # dataset.ordered_features is an ordered list of tokens.
    # TODO(rpryzant) -- deprecate this as the data is redundant.
    self.assertEqual(self.dataset.ordered_features, ['<unk>', 'is', 'not'])

    # The system should have written a vocab.
    self.assertEqual(self.dataset.vocab,
                     self.config_mock.working_dir + '/freq_vocab.txt')

  def test_y_batch_continuous_targets(self):
    """Ensures the dataset can retrieve batches of continuous Y targets."""
    self.dataset.set_active_split(self.config_mock.train_suffix)

    # Verify we can pull out the continuous targets.
    self.assertTrue(
        np.array_equal(
            self.dataset.y_batch('continuous_target'), np.array([1.0, 1.1])))
    # We can also pull out a subset of the data.
    self.assertEqual(
        self.dataset.y_batch('continuous_target', start=0, end=1), 1.0)
    self.assertEqual(self.dataset.y_batch('continuous_target', start=1), 1.1)
    # We can also pull out confound data.
    self.assertEqual(self.dataset.y_batch('continuous_confound', start=1), 3.2)

    # We can pull out data from any split, not just train.
    self.dataset.set_active_split(self.config_mock.test_suffix)
    self.assertEqual(self.dataset.y_batch('continuous_target'), 1.3)
    self.assertEqual(self.dataset.y_batch('continuous_confound'), 2.4)

  def test_y_batch_categorical_targets(self):
    """Ensures the dataset can retrieve batches of categorical Y targets."""
    self.dataset.set_active_split(self.config_mock.train_suffix)

    # We can pull out all of the data for a target.
    self.assertTrue(
        np.array_equal(
            self.dataset.y_batch('categorical_target', target_level='a'),
            np.array([1.0, 1.0])))
    # The vectors we pull out tell us whether an example belongs to the
    # target class. None of the train examples belong to class `b` so the
    # returned vector is all 0's.
    self.assertTrue(
        np.array_equal(
            self.dataset.y_batch('categorical_target', target_level='b'),
            np.array([0.0, 0.0])))
    # We can also pull out data for the confounds.
    self.assertTrue(
        np.array_equal(
            self.dataset.y_batch('categorical_confound', target_level='_1'),
            np.array([1.0, 1.0])))

    # We can pull out a subset of the data.
    self.assertEqual(
        self.dataset.y_batch(
            'categorical_target', target_level='a', start=0, end=1), 1.0)

    # We can pull out data for any split.
    self.dataset.set_active_split(self.config_mock.dev_suffix)

    self.assertEqual(
        self.dataset.y_batch('categorical_target', target_level='b'), 1.0)
    self.assertEqual(
        self.dataset.y_batch('categorical_confound', target_level='_1'), 0.0)
    self.assertEqual(
        self.dataset.y_batch('categorical_confound', target_level='_2'), 1.0)

  def test_text_x_batch(self):
    """Ensures the dataset can retrieve text covariates from the data."""
    self.dataset.set_active_split(self.config_mock.train_suffix)

    # All of the train examples contain the complete vocabulary.
    x, x_feature_names = self.dataset.text_x_batch()
    self.assertTrue(np.array_equal(x, np.array([[1, 1, 1], [1, 1, 1]])))
    self.assertEqual(x_feature_names, ['<unk>', 'is', 'not'])

    # The test example is missing `not`.
    self.dataset.set_active_split(self.config_mock.test_suffix)
    x, x_feature_names = self.dataset.text_x_batch()
    self.assertTrue(np.array_equal(x, np.array([[1, 1, 0]])))
    self.assertEqual(x_feature_names, ['<unk>', 'is', 'not'])

    x, x_feature_names = self.dataset.text_x_batch(feature_subset=['not'])
    self.assertEqual(x, 0)
    self.assertEqual(x_feature_names, ['not'])

  def test_nontext_x_batch(self):
    """Ensures the dataset can retrieve confound covariates from the data."""

    self.dataset.set_active_split(self.config_mock.train_suffix)

    x, x_feature_names = self.dataset.nontext_x_batch(
        ['categorical_confound', 'continuous_confound'])
    self.assertTrue(
        np.array_equal(x, np.array([[1, 0, 0, 3.1], [1, 0, 0, 3.2]])))
    self.assertEqual(x_feature_names, [
        'categorical_confound|_1', 'categorical_confound|_2',
        'categorical_confound|_3', 'continuous_confound'
    ])

    x, x_feature_names = self.dataset.nontext_x_batch(['categorical_confound'])
    self.assertTrue(np.array_equal(x, np.array([[1, 0, 0], [1, 0, 0]])))
    self.assertEqual(x_feature_names, [
        'categorical_confound|_1', 'categorical_confound|_2',
        'categorical_confound|_3'
    ])

    x, x_feature_names = self.dataset.nontext_x_batch(
        ['continuous_confound'], start=1)
    self.assertTrue(np.array_equal(x, np.array([[3.2]])))
    self.assertEqual(x_feature_names, ['continuous_confound'])

    self.dataset.set_active_split(self.config_mock.test_suffix)
    x, x_feature_names = self.dataset.nontext_x_batch(
        ['categorical_confound', 'continuous_confound'])
    self.assertTrue(np.array_equal(x, np.array([[0, 0, 1, 2.4]])))
    self.assertEqual(x_feature_names, [
        'categorical_confound|_1', 'categorical_confound|_2',
        'categorical_confound|_3', 'continuous_confound'
    ])

  def test_make_tf_iterators(self):
    """Tests data provider operations."""
    self.dataset.set_active_split(self.config_mock.train_suffix)

    iterator = self.dataset.make_tf_iterators(batch_size=2)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())

      # Verify the inputs.
      sess.run(iterator['initializer'])
      inputs = sess.run(iterator['input'])
      tokens, ids, lengths = inputs

      # The inputs should have been cut of at max_seq_len (3).
      self.assertTrue(
          all([length == self.config_mock.max_seq_len for length in lengths]))

      # Input tokens should be the same as the true tokens.
      self.assertEqual(list(tokens[0]), list(tokens[1]))
      self.assertEqual(list(tokens[0]), ['is', 'not', 'be'])

      # Tokens should be mapped to their proper IDs.
      # I.e. `be` should be mapped to `unk` (0).
      self.assertEqual(list(ids[0]), [1, 2, 0])

      # Now verify the targets.
      sess.run(iterator['initializer'])
      continuous_target = sess.run(iterator['continuous_target'])
      sess.run(iterator['initializer'])
      categorical_target = sess.run(iterator['categorical_target'])

      # Round floats due to precision loss inside of TensorFlow.
      self.assertEqual([round(x, 1) for x in continuous_target], [1.0, 1.1])
      # The category for this outcome (`_1`) has ID 0.
      self.assertEqual(self.dataset.class_to_id_map['categorical_target']['a'],
                       0)
      self.assertEqual(list(categorical_target), [0, 0])

      # Last, verify the confounds.
      sess.run(iterator['initializer'])
      continuous_target = sess.run(iterator['continuous_confound'])
      sess.run(iterator['initializer'])
      categorical_target = sess.run(iterator['categorical_confound'])

      # Round floats due to precision loss inside of TensorFlow.
      self.assertEqual([round(x, 1) for x in continuous_target], [3.1, 3.2])
      # The category for this outcome (`_1`) has ID 0.
      self.assertEqual(
          self.dataset.class_to_id_map['categorical_confound']['_1'], 0)
      self.assertEqual(list(categorical_target), [0, 0])

  def test_make_tf_iterators_ngram(self):
    """Tests ngram data provider operations."""
    # Make a new dataset with bigram features.
    bigram_dataset, bigram_config_mock = make_dataset_and_config(
        os.path.join(self.tmp_dir, 'bigrams'), bigrams=True)

    bigram_dataset.set_active_split(bigram_config_mock.train_suffix)

    # Run the iterator normally and make sure we get our bigrams back.
    iterator = bigram_dataset.make_tf_iterators(batch_size=2, undo_ngrams=True)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())
      sess.run(iterator['initializer'])
      inputs = sess.run(iterator['input'])
      tokens, _, _ = inputs
      self.assertEqual(list(tokens[0]), list(tokens[1]))
      self.assertEqual(list(tokens[0]), ['is', 'not', 'be'])

    # Now make a new iterator with the `undo_ngrams` flag set to true,
    # and make sure that we've ignored the ngrams.
    iterator = bigram_dataset.make_tf_iterators(batch_size=2, undo_ngrams=True)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())
      sess.run(iterator['initializer'])
      inputs = sess.run(iterator['input'])
      tokens, _, _ = inputs
      # Even though the raw input data are bigrams, the iterator should
      # act as if they were never ngrammed.
      self.assertEqual(list(tokens[0]), list(tokens[1]))
      self.assertEqual(list(tokens[0]), ['is', 'not', 'be'])


if __name__ == '__main__':
  googletest.main()
