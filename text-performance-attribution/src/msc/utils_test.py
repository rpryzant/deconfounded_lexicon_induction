"""Tests for src.msc.utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest

import random
from absl.testing import absltest
import numpy as np
import pytest

import sys; sys.path.appen('../..')
from src.msc import utils


class UtilsTest(unittest.TestCase):

  def setUp(self):
    random.seed(1)

    self.tmp_dir = absltest.get_default_test_tmpdir()

  def test_split_in_three(self):
    """Tests breaking a file into train/test/dev splits."""

    # Create a test file for splitting.
    input_file = self.tmp_dir + '/split_input'
    output_train = self.tmp_dir + '/split_train'
    output_dev = self.tmp_dir + '/split_dev'
    output_test = self.tmp_dir + '/split_test'

    with open(input_file, 'w') as f:
      f.write('\n'.join([str(x) for x in range(100)]))

    # Now split this file into three parts.
    utils.split_in_three(
        input_file_path=input_file,
        dev_size=10,
        test_size=15,
        train_out_path=output_train,
        dev_out_path=output_dev,
        test_out_path=output_test)

    # Verify the length of each file.
    train_lines = open(output_train).readlines()
    dev_lines = open(output_dev).readlines()
    test_lines = open(output_test).readlines()

    self.assertEqual(len(train_lines), 100 - 10 - 15)
    self.assertEqual(len(dev_lines), 10)
    self.assertEqual(len(test_lines), 15)

    # Verify that the contents of each file don't overlap.
    self.assertEqual(len(set(train_lines) & set(dev_lines)), 0)
    self.assertEqual(len(set(train_lines) & set(test_lines)), 0)
    self.assertEqual(len(set(dev_lines) & set(test_lines)), 0)

  def test_standardize(self):
    """Tests array standardization."""
    normal_input = np.array([1.1, 4.2, 0.1, 0.2])
    self.assertEqual([round(x, 2) for x in utils.standardize(normal_input)],
                     [-0.18, 1.68, -0.78, -0.72])

    normal_input = np.array([0.0, 400, 3, -1000])
    self.assertEqual([round(x, 2) for x in utils.standardize(normal_input)],
                     [0.29, 1.06, 0.29, -1.64])

    nested_input = np.array([[0.0, 400, 3, -1000]])
    self.assertEqual([round(x, 2) for x in utils.standardize(nested_input)[0]],
                     [0.29, 1.06, 0.29, -1.64])

    matrix_input = np.array([[1, 3], [-1, 2]])
    self.assertEqual(
        [[round(x, 2) for x in row] for row in utils.standardize(matrix_input)],
        [[-0.17, 1.18], [-1.52, 0.51]])

    zero_input = np.array([0, 0, 0])
    self.assertEqual(list(utils.standardize(zero_input)), [0, 0, 0])

    no_std_input = np.array([1, 1, 1])
    self.assertEqual(list(utils.standardize(no_std_input)), [0, 0, 0])

  def test_is_number(self):
    """Tests number checking."""
    self.assertTrue(utils.is_number('1'))
    self.assertTrue(utils.is_number(2))
    self.assertTrue(utils.is_number('1.3'))
    self.assertTrue(utils.is_number('-2.3'))
    self.assertTrue(utils.is_number(1e3))
    self.assertTrue(utils.is_number('1e3'))
    x = 3
    self.assertTrue(utils.is_number(x))
    x = '3'
    self.assertTrue(utils.is_number(x))

    self.assertFalse(utils.is_number('foo'))
    self.assertFalse(utils.is_number('seven'))
    self.assertFalse(utils.is_number([]))
    x = 'i promise im a number'
    self.assertFalse(utils.is_number(x))

  def test_pickle_depickle(self):
    """Tests object saving/loading utilities."""

    def test_object(obj, equality_fn=None):
      obj_path = self.tmp_dir + '/obj'
      utils.pickle(obj, obj_path)
      loaded = utils.depickle(obj_path)
      if equality_fn is None:
        self.assertEqual(obj, loaded)
      else:
        self.assertTrue(equality_fn(obj, loaded))

    test_object(range(10))
    test_object({'a': 1})
    test_object(np.array([1, 2, 3]), np.array_equal)

    def check_nested_np_array(a, b):
      for a_key, b_key in zip(sorted(a.keys()), sorted(b.keys())):
        if not np.array_equal(a[a_key], b[b_key]):
          return False
      return True

    test_object({'a': np.array([1, 2, 3])}, check_nested_np_array)

  def test_percentile(self):
    """Tests percentile calculations."""
    arr = range(100)
    # 70 is the number at the 70th percentile of arr.
    number_at_percentile = utils.percentile(arr, 0.7)
    self.assertEqual(number_at_percentile, 70)

    arr = range(150)
    number_at_percentile = utils.percentile(arr, 0.7)
    self.assertEqual(number_at_percentile, 105)

    arr = range(150)
    number_at_percentile = utils.percentile(arr, 0.0)
    self.assertEqual(number_at_percentile, 0)

    arr = range(150)
    number_at_percentile = utils.percentile(arr, 0.999)
    self.assertEqual(number_at_percentile, 149)

    with self.assertRaises(IndexError):
      number_at_percentile = utils.percentile(arr, 1.0)

    with self.assertRaises(AssertionError):
      utils.percentile(arr, -0.1)

  def test_rank_threshold(self):
    """Tests rank thresholding."""
    # 69 is the number at the 30th percentile of arr.
    arr = range(100)
    number_at_percentile = utils.rank_threshold(arr, 30)
    self.assertEqual(number_at_percentile, 69)

    arr = range(150)
    number_at_percentile = utils.rank_threshold(arr, 45)
    self.assertEqual(number_at_percentile, 104)

    with self.assertRaises(AssertionError):
      utils.rank_threshold(arr, -10)


if __name__ == '__main__':
  googletest.main()
