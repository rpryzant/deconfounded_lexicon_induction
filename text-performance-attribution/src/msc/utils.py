"""Static utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import cPickle
import getpass
import numpy as np
import yaml
import os

# This placeholder occurs in file paths of the sample_config.yaml which is
# for integration testing. This lets the system point itself at local data on
# your workstation.
_USER_PLACEHOLDER = '<USER>'

# Global constants for variable types.
# These are used throughout the system to type-check variable specs.
CONTINUOUS = 'continuous'
CATEGORICAL = 'categorical'
TEXT = 'text'


def split_in_three(input_file_path, dev_size, test_size, train_out_path,
                   dev_out_path, test_out_path):
  """Breaks a file into train/dev/test splits.

  This method doesn't do anything if the split files already exist.

  Args:
    input_file_path: string, the path which is to be split.
    dev_size: int, the size of the dev split.
    test_size: int, the size of the test split.
    train_out_path: string, filepath for the train split.
    dev_out_path: string, filepath for the dev split.
    test_out_path: string, filepath for the test split.
  """
  if os.path.exists(train_out_path) and os.path.exists(
      dev_out_path) and os.path.exists(test_out_path):
    return

  with open(input_file_path, 'r') as in_file, open(
      train_out_path,
      'a') as train_out, open(dev_out_path, 'a') as dev_out, open(
          test_out_path, 'a') as test_out:
    for i, row in enumerate(in_file):
      if i < dev_size:
        dev_out.write(row)
      elif i < dev_size + test_size:
        test_out.write(row)
      else:
        train_out.write(row)


def standardize(arr):
  """Standardizes a np array arr: subtracts off the mean and divides by std."""
  return (arr - np.mean(arr)) / (np.std(arr) or 1e-3)


def is_number(x):
  """Tests whether a variable (e.g. '12') contains a number."""
  try:
    float(x)
    return True
  except (ValueError, TypeError):
    return False


def pickle(obj, path):
  """Pickles an object and dumps it into the specified `path`."""
  with open(path, 'w') as f:
    cPickle.dump(obj, f)


def depickle(path):
  """Loads a pickled object from the specified `path`."""
  with open(path, 'r') as f:
    return cPickle.load(f)


def percentile(arr, threshold):
  """Finds the number at threshold^th percentile of `arr`."""
  assert threshold >= 0

  idx = int(threshold * len(arr))
  return sorted(arr)[idx]


def rank_threshold(arr, rank, largest=True):
  """Finds the percentile threshold for the `rank` largest numbers in `arr`."""
  assert rank >= 0

  if largest:
    return sorted(arr)[::-1][rank]
  else:
    return sorted(arr)[rank]


def file_len(path):
  """Finds the length of the file at `path`."""
  return sum(1 for _ in open(path))


def load_yaml_config(filename):
  """Loads YAML config into a named tuple."""
  config_dict = yaml.load(open(filename).read())
  # Insert username into path-dependent config fields if needed.
  username = getpass.getuser()
  if _USER_PLACEHOLDER in config_dict['working_dir']:
    config_dict['working_dir'] = config_dict['working_dir'].replace(
        _USER_PLACEHOLDER, username)
  if _USER_PLACEHOLDER in config_dict['data_path']:
    config_dict['data_path'] = config_dict['data_path'].replace(
        _USER_PLACEHOLDER, username)

  return namedtuple('config', config_dict.keys())(**config_dict)


def write_config(config, path):
  """Writes a config object to path."""
  yaml_str = yaml.dump(dict(config._asdict()), default_flow_style=False)
  with open(path, 'w') as f:
    f.write(yaml_str)
