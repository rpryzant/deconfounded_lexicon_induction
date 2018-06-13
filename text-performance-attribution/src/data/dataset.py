"""Logic for managing datsets and iterating over them."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import defaultdict
import os
import time
import numpy as np
from scipy import sparse
import tensorflow as tf

import src.msc.utils as utils

# Hardcode the unk symbol's ID.
# This assumes unk is at top of vocab file.
# We enforce this assumption in _check_vocab().
_UNK_ID = 0
# Variable names will have this instead of whitespace.
_UNDERSCORE = '_'


class Dataset(object):
  """Class for managing datsets and iterating over them."""

  def __init__(self, config, base_dir):
    """Initializes a Dataset object.

    Upon initialization, the object will
      -create new datafiles, one per outcome variable
      -generate a vocab
      -parse the data into a series of np arrays (one per variable)

    Args:
      config: NamedTuple, a config.yaml file that's been parsed into an object.
      base_dir: string, a directory where the system will write outputs.
    """
    self.config = config
    self.base_dir = base_dir

    # The train set is the default split.
    self.current_split = config.train_suffix

    assert self.config.data_spec[0]['type'] == utils.TEXT, (
        'Text input must be the first column of the input '
        'data!')

    # Create individual files for each variable.
    # Create dictionaries for mapping {split: {variable name: filepath} }.
    # We have to have these per-variable files to make tf.Iterators
    # in dataset.Dataset.make_tf_iterators()
    print('DATASET: making splits...')
    self.data_files, self.split_sizes, self.whole_data_files = self._cut_data()

    # Create IDs for each level of a categorical variable, and map
    # variable names to their level.
    # class_to_id_map: {variable name: {'class': index}  }
    self.class_to_id_map, self.id_to_class_map = self._get_categorical_tables()

    # Generate a vocabulary.
    # self.vocab = filepath to vocab file
    if self.config.vocab['vocab_file'] is None:
      start = time.time()
      input_seqs = self.whole_data_files[self.input_varname()]
      self.vocab = self._gen_vocab(input_seqs)
    else:
      self.vocab = self.config.vocab['vocab_file']

    # Create vocab maps that represent the vocabulary in 3 ways:
    # features: vocab --> index
    # ids_to_feat: index --> vocab
    # ordered_feat: list of v1, v2... ordered by index
    self.vocab_size = self._check_vocab(self.vocab)
    self.features = {
        v.strip(): i for i, v in enumerate(open(self.vocab))
    }
    self.ids_to_features = {i: f for f, i in self.features.items()}
    self.ordered_features = [
        self.ids_to_features[i] for i in range(self.vocab_size)
    ]

    # Parse the data into the form we need for training
    start = time.time()
    np_path = os.path.join(config.working_dir, 'np_data.pkl')
    if not os.path.exists(np_path):
      print('DATASET: parsing data into np arrays...')
      self.np_data = self._get_np_data()
      utils.pickle(self.np_data, np_path)
    else:
      print('DATASET: restoring np_arrays from %s', np_path)
      self.np_data = utils.depickle(np_path)
    print('\tdone, took %.2fs', time.time() - start)

  def datafile_to_np(self, datafile, feature_type, feature_id_map=None):
    """Parses a data file into an np array.

    Args:
      datafile: string, path to a text file with one example per line.
      feature_type: string, one of [`text`, `categorical`, `continuous`].
        This specifies the kind of data the input `datafile` holds.
      feature_id_map: dictionary {'class': index}, for categorical variables the
        returned array will have one-hot rows whose
        values correspond to the values of feature_id_map.

    Returns:
      a sparse.csr matrix representing the data in `datafile`.
    """
    num_examples = utils.file_len(datafile)
    num_features = len(feature_id_map) if feature_id_map else 1
    out = np.zeros((num_examples, num_features))

    for i, line in enumerate(open(datafile)):
      line = line.strip()
      if feature_type == utils.TEXT:
        for feature in line.split()[:self.config.max_seq_len]:
          out[i][feature_id_map.get(feature, _UNK_ID)] = 1
      elif feature_type == utils.CATEGORICAL:
        out[i][feature_id_map.get(line.replace(' ', _UNDERSCORE), _UNK_ID)] = 1
      else:
        out[i][0] = float(line)
    return sparse.csr_matrix(out)

  def _get_np_data(self):
    """Converts all of the input data to np arrays.

    Note that if the array is for a categorical variable, its columns
    will be indexed by the IDs in self.class_to_id_map

    Args:
      Nothing.

    Returns:
      np_data: dict, this is a dictionary with shape
        {split name {variable name: np array } }, where each np.array
        holds the data for that split and variable.
    """
    np_data = defaultdict(dict)
    for split, variables in self.data_files.items():
      for varname, filepath in variables.items():
        var = self.get_variable(varname)
        # Ignore skipped variables.
        if var['type'] != utils.TEXT and var['skip']:
          continue
        if var['type'] == utils.CONTINUOUS:
          np_data[split][varname] = self.datafile_to_np(filepath,
                                                        utils.CONTINUOUS)
        elif var['type'] == utils.TEXT:
          np_data[split][varname] = self.datafile_to_np(
              filepath, utils.TEXT, feature_id_map=self.features)
        else:
          np_data[split][varname] = self.datafile_to_np(
              filepath,
              utils.CATEGORICAL,
              feature_id_map=self.class_to_id_map[varname])
    return np_data

  def _get_categorical_tables(self):
    """Generates IDs for the levels of each categorical variable.

    Args:
      Nothing

    Returns:
      class_to_id_map: dict, a nested map from variable => level => ID.
      id_to_class_map: dict, a nested map from variable => ID => level.
    """
    class_to_id_map = defaultdict(dict)
    id_to_class_map = defaultdict(dict)
    for variable in self.config.data_spec[1:]:
      # Ignore skipped variables.
      if variable['type'] != utils.CATEGORICAL or variable['skip']:
        continue
      i = 0
      var_filename = self.whole_data_files[variable['name']]

      levels = set(open(var_filename).readlines())
      for level in sorted(list(levels)):
        level = level.strip()

        if level in class_to_id_map[variable['name']]:
          continue
        # Whitespaces are not allowed in level names.
        level = level.replace(' ', _UNDERSCORE)
        class_to_id_map[variable['name']][level] = i
        id_to_class_map[variable['name']][i] = level
        i += 1
    return class_to_id_map, id_to_class_map

  def y_batch(self, target_name, target_level=None, start=0, end=None):
    """Retrieves a batch of labels from the dataset.

    Args:
      target_name: string, the name of the variable which is to be retrieved.
      target_level: string, the categorical level which is to be retrieved.
        If supplied, the retrieved variable is assumed to be categorical.
      start: int, the index where the batch should start.
      end: int, the index where the batch should end. If this isn't specified,
        the end will be set to the dataset's end.

    Returns:
      y: np array, a batch of labels where the variable named `target_name`. If
        `target_level` was specified, the returned `y` is equivilant to
        a binary variable which has 1's where the example had that level.
    """
    if end is None:
      end = self.split_sizes[self.current_split]
    else:
      end = min(end, self.split_sizes[self.current_split])

    # Pull out the target for this batch of examples.
    y = self.np_data[self.current_split][target_name][start:end].toarray()

    if target_level is not None:
      target_col = self.class_to_id_map[target_name][target_level]
      y = y[:, target_col]

    y = np.squeeze(y)
    return y

  def text_x_batch(self, feature_subset=None, start=0, end=None):
    """Retrieves a batch of text covariates from the dataset.

    Args:
      feature_subset: list(string)
        A list of tokens to restrict the retrieval with.
      start: int
        Batch start point.
      end: int
        Batch end point. Will be set to the dataset's end if not provided.

    Returns:
      X: np array
        A matrix with bag-of-words occurrence counts on each row, and
        features on each column.
      X_features: list(string)
        The names of each of the columns of X.
    """
    if end is None:
      end = self.split_sizes[self.current_split]
    else:
      end = min(end, self.split_sizes[self.current_split])

    # Start with all the text features for desired batch.
    x = self.np_data[self.current_split][self.input_varname()][start:
                                                               end].toarray()
    x_features = self.ordered_features

    if feature_subset is not None:
      assert feature_subset
      retain_indices = [self.features[f] for f in feature_subset]
      x = x[:, retain_indices]
      x_features = feature_subset

    return x, x_features

  def nontext_x_batch(self, features, start=0, end=None):
    """Retrieves a batch of non-textual covariates from the dataset.

    Args:
      features: list(string), a list of tokens to restrict the retrieval with.
      start: int, the batch start point.
      end: int, the batch end point, which defaults to the dataset's end.

    Returns:
      x: np array, a matrix with one-hot vectors on each row (for categorical
        features) or scalars (for continuous features).
      x_features: list(string), the names of each of the columns of X.
    """
    if end is None:
      end = self.split_sizes[self.current_split]
    else:
      end = min(end, self.split_sizes[self.current_split])

    x_features = []
    cols = []
    for varname in sorted(features):
      one_hots = self.np_data[self.current_split][varname][start:end].toarray()
      if self.get_variable(varname)['type'] == utils.CATEGORICAL:
        for level, idx in sorted(self.class_to_id_map[varname].items()):
          cols.append(one_hots[:, idx])
          x_features.append('%s|%s' % (varname, level))
      else:
        cols.append(one_hots)
        x_features.append(varname)

    x = np.column_stack(cols)
    return x, x_features

  def input_varname(self):
    """Gets the name of the text input variable."""
    return self.config.data_spec[0]['name']

  def get_variable(self, varname):
    """Gets the specification (from config.data_spec) for a variable name."""
    return next((v for v in self.config.data_spec if v['name'] == varname))

  def set_active_split(self, split):
    """Points the dataset towards a split."""
    self.current_split = split

  def get_tokenized_input(self):
    """Returns a tokenized version of the input text data."""
    return [
        line.strip().split() for line in open(self.data_files[
            self.current_split][self.input_varname()])
    ]

  def data_for_variable(self, variable):
    """Returns the raw, unparsed data for a variable."""
    eval_fn = str if variable['type'] == utils.CATEGORICAL else float
    return [
        eval_fn(x.strip().replace(' ', _UNDERSCORE)) for x in open(
            self.data_files[self.current_split][variable['name']])
    ]

  def num_levels(self, name):
    """Gets the number of levels for a categorical variable."""
    return len(self.class_to_id_map[name])

  def _check_vocab(self, vocab_file):
    """Checks a vocab to make sure it doesn't have dups and starts with unk."""
    assert os.path.exists(
        vocab_file), 'The vocab file %s does not exist' % vocab_file

    lines = [x.strip() for x in open(vocab_file)]

    if len(lines) != len(set(lines)):
      print('DATASET: vocab %s contains dups! fixing.', vocab_file)
      unk = self.config.unk
      os.system('rm %s' % vocab_file)
      s = unk + '\n' + '\n'.join([x for x in set(lines) if x != unk])
      with open(vocab_file, 'w') as f:
        f.write(s)
      # Re-read the vocab so that we can get the new length.
      lines = [x.strip() for x in open(vocab_file)]

    assert lines[0] == self.config.unk, 'The first words in %s is not %s' % (
        vocab_file)

    return len(lines)

  def _gen_vocab(self, text_file):
    """Generates a vocab of the top K most frequent tokens from a text file.

    Args:
      text_file: string, a path to the text file for which we are
        generating a vocab.

    Returns:
      vocab_path: string, a path to the newly generated vocabulary. If
        config.vocab.vocab_file is specified,
        this path will point to that file (after it's been validated). If
        the file path is None, this path will point to a new vocabulary which
        contains the top config.vocab.top_n most frequent tokens.
    """
    vocab_path = os.path.join(self.base_dir, 'freq_vocab.txt')
    if not os.path.exists(vocab_path):
      print('DATASET: making vocab of %d tokens..',
                   self.config.vocab['top_n'])
      start = time.time()
      word_ctr = Counter(open(text_file).read().split())
      vocab = [
          word for word, _ in word_ctr.most_common(self.config.vocab['top_n'])
      ]
      with open(vocab_path, 'w') as f:
        f.write('\n'.join(vocab))
      print('\tdone. took %.fs', time.time() - start)
    else:
      print('DATASET: restoring vocab from %s', vocab_path)
      vocab = [x.strip() for x in open(vocab_path).readlines()
              ][1:]  # unk is 0th element.

    vocab = [self.config.unk] + vocab

    with open(vocab_path, 'w') as f:
      f.write('\n'.join(vocab))

    return vocab_path

  def _cut_data(self):
    """Beaks a TSV into one file per column, returns paths for those files."""
    c = self.config

    split_sizes = {}

    data_prefix = c.data_path
    variable_paths = defaultdict(dict)
    whole_data_paths = {}
    for split_suffix in [c.train_suffix, c.dev_suffix, c.test_suffix]:
      in_file = data_prefix + split_suffix
      assert os.path.exists(in_file), 'Split %s doesnt exist' % in_file

      split_sizes[split_suffix] = utils.file_len(in_file)

      for i, variable in enumerate(c.data_spec):
        if i > 0 and variable['skip']:
          continue
        variable_path = data_prefix + '.' + variable['name'] + split_suffix
        variable_path_nosplit = data_prefix + '.' + variable['name']

        variable_paths[split_suffix][variable['name']] = variable_path
        whole_data_paths[variable['name']] = variable_path_nosplit

        if not os.path.exists(variable_path):
          with open(in_file) as in_handler, open(
              variable_path, 'a') as out_handler:
            for line in in_handler:
              out_handler.write(line.strip().split('\t')[i] + '\n')

        # os.system('cat %s | cut -f%d > %s' % (in_file, i + 1, variable_path))
        if not os.path.exists(variable_path_nosplit):
          with open(data_prefix) as in_handler, open(
              variable_path_nosplit, 'a') as out_handler:
            for line in in_handler:
              out_handler.write(line.strip().split('\t')[i] + '\n')

        #  os.system('cat %s | cut -f%d > %s' % (data_prefix, i + 1,
        #                                       variable_path_nosplit))

    return variable_paths, split_sizes, whole_data_paths

  def make_tf_iterators(self, batch_size, undo_ngrams=False):
    """Creates TensorFlow tf.Dataset iterators from the data.

    Args:
      batch_size: int, the batch size to use.
      undo_ngrams: bool, if true, then the text data are in fact ngrams
        joined by `_`, and we need to "undo" the ngrams and iterate as if
        the tokens were never joined into ngrams. We assume that the ngrams
        are in the same order as the original sentence, i.e.
          "Frugle Google Boogle" => "Frugle_Google", "Google_Boogle"
        We do not make any assumptions about `n`, the ngram sizes, other than
        that appear in order, i.e. if we have ngram sizes of 2 and 3
        then we would expect:
           "Frugle Google Boogle" => "Frugle_Google", "Google_Boogle",
                                     "Frugle_Google_Boogle"

    Returns:
      out: dict, a dictionary mapping each variable to a tf iterator's
        placeholder, along with the special key 'initializer' which maps to
        the initializer for this iterator.
    """
    # Break up all of the ngrams and use their tokens as vocab features.
    if undo_ngrams:
      nested_unigrams = [x.strip().split('_') for x in open(self.vocab)]
      unigrams = list(
          set([f for broken_ngram in nested_unigrams for f in broken_ngram]))
      feature_list_tensor = tf.constant(unigrams)
      vocab_table = tf.contrib.lookup.index_table_from_tensor(
          feature_list_tensor, default_value=_UNK_ID)
    # Simply use the original vocab file.
    else:
      vocab_table = tf.contrib.lookup.index_table_from_file(
          self.vocab, default_value=_UNK_ID)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(self.config.eos)), tf.int32)

    def text_dataset(filepath):
      """Make a tf.Dataset from a file with text sequences on each line."""
      # Create a copy of the data in `filepath` with the same content and
      # ordering but without any ngrams.
      if undo_ngrams:
        unigram_path = os.path.join(self.config.working_dir,
                                    'unigrams.%s' % os.path.basename(filepath))
        with open(unigram_path, 'a') as out_file, open(
            filepath, 'r') as in_file:
          for line in in_file:
            parts = line.strip().split()
            # Remove all non-first-order ngrams, and only keep the first joined
            # token from each ngram. The ngrams are a sliding window over the
            # source so this will recover the original content.
            parts = [x for x in parts if x.count('_') == parts[0].count('_')]
            unigrams = [ngram.split('_')[0] for ngram in parts]
            # Corner case: ngrams never start with the last n-1 tokens in a seq.
            unigrams += parts[-1].split('_')[1:]

            out_file.write(' '.join(unigrams) + '\n')

        dataset = tf.data.TextLineDataset(unigram_path)

      # We do not want to undo any ngrams, so use the provided path as-is.
      else:
        dataset = tf.data.TextLineDataset(filepath)

      # Break sentences into tokens.
      dataset = dataset.map(lambda txt: tf.string_split([txt]).values)
      # Convert to ids.
      dataset = dataset.map(
          lambda txt: (txt, tf.cast(vocab_table.lookup(txt), tf.int32)))

      # Now cut off at max_seq_len.
      maxlen = self.config.max_seq_len
      dataset = dataset.map(lambda txt, ids: (txt[:maxlen], ids[:maxlen]))

      # Add lengths of each sequence.
      dataset = dataset.map(lambda txt, ids: (txt, ids, tf.size(ids)))

      return dataset

    def continuous_dataset(filepath):
      """Makes a tf.Dataset from a file of numbers."""
      dataset = tf.data.TextLineDataset(filepath)
      # Append 0 to start in case there's blank rows.
      dataset = dataset.map(tf.string_to_number)
      return dataset

    def categorical_dataset(filepath, variable_name):
      """Makes a tf.Dataset from a file of categorical values."""
      dataset = tf.data.TextLineDataset(filepath)

      classes_ids = self.class_to_id_map[variable_name]
      # Include spaces here because they're in the raw data but not
      #    self.class_to_id_map
      dict_with_spaces = {
          level.replace(_UNDERSCORE, ' '): idx
          for level, idx in classes_ids.items()
      }
      dict_with_spaces.update(classes_ids)
      class_lookup_table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(
              keys=dict_with_spaces.keys(),
              values=dict_with_spaces.values(),
              key_dtype=tf.string,
              value_dtype=tf.int32), _UNK_ID)
      # pylint: disable=unnecessary-lambda
      dataset = dataset.map(lambda x: class_lookup_table.lookup(x))
      # pylint: disable=unnecessary-lambda
      return dataset

    def batch_up(datset):
      """Buckets, batches, and pads a tf.Dataset."""
      # The first element is (text, text, text len), followed by all other vars.
      num_variables = len(
          [v for v in self.config.data_spec[1:] if not v['skip']])
      padded_shapes = tuple([(tf.TensorShape([None]), tf.TensorShape([None]),
                              tf.TensorShape([]))] +
                            [tf.TensorShape([]) for _ in range(num_variables)])

      # Pad text with eos, otherwise 0 (means unused).
      padding_values = [(self.config.eos, eos_id, 0)]

      for var in self.config.data_spec[1:]:
        # Ignore skipped variables.
        if var['skip']:
          continue

        if var['type'] == utils.CATEGORICAL:
          padding_values.append(0)
        else:
          padding_values.append(0.0)

      padding_values = tuple(padding_values)

      return datset.padded_batch(
          batch_size,
          padded_shapes=padded_shapes,
          padding_values=padding_values)

    datasets = []
    for i, variable in enumerate(self.config.data_spec):
      if i > 0 and variable['skip']:
        continue

      data_file = self.data_files[self.current_split][variable['name']]
      if variable['type'] == utils.TEXT:
        dataset = text_dataset(data_file)
      elif variable['type'] == utils.CONTINUOUS:
        dataset = continuous_dataset(data_file)
      else:
        dataset = categorical_dataset(data_file, variable['name'])
      datasets.append(dataset)

    dataset = tf.data.Dataset.zip(tuple(datasets))
    dataset = batch_up(dataset)

    out = {}
    iterator = dataset.make_initializable_iterator()
    data_spec = [v for v in self.config.data_spec if not v.get('skip', False)]
    placeholders = iterator.get_next()
    for i, (placeholder, variable) in enumerate(zip(placeholders, data_spec)):
      out[variable['name']] = placeholder
    out['initializer'] = iterator.initializer

    return out
