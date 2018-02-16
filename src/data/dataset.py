""" data management and iteration """

import os
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
import pandas as pd
from tensorflow.python.ops import lookup_ops
import tensorflow as tf
import time
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import src.msc.utils as utils
import numpy as np
from scipy import sparse
import src.feature_selectors.odds_ratio as odds_ratio
import src.feature_selectors.mutual_information as mutual_information

# assumes unk is at top of vocab file but we are enforcing that in _check_vocab()
UNK_ID = 0



class Dataset(object):
    """ this objectis VERY IMPORTANT!!
    
        it is the hub for all data manipulation logic and knowledge.
        this and the config should contain basically all the information
            needed to run an experiment from start to finish
    """

    def __init__(self, config, base_dir):
        self.config = config
        self.base_dir = base_dir
        assert self.config.data_spec[0]['type'] == 'text', \
            'text input must be first element of data spec!'

        # this is {train/val/test: {variable name: filepath with just that variable on each line}  }
        print 'DATASET: making splits...'
        self.data_files, self.split_sizes, self.whole_data_files = self._cut_data()

        # class_to_id_map: {variable name: {'class': index}  }  for each categorical variable
        self.class_to_id_map, self.id_to_class_map = self._get_categorical_tables()

        # self.vocab = filepath to vocab file
        if self.config.vocab['vocab_file'] is None:
            start = time.time()
            input_seqs = self.whole_data_files[self.input_varname()]
            self.vocab = self._gen_vocab(input_seqs)
        else:
            self.vocab = self.config.vocab['vocab_file']
        # represent vocab in 3 ways:
            # features: vocab --> index
            # ids_to_feat: index --> vocab
            # ordered_feat: list of v1, v2... ordered by index 
        self.vocab_size = self._check_vocab(self.vocab)
        self.features = {v.strip(): i for i, v in enumerate(open(self.vocab))}
        self.ids_to_features = {i: f for f, i in self.features.items()}
        self.ordered_features = [self.ids_to_features[i] for i in range(self.vocab_size)]

        # build {split {variable: np array with that var's data (columns indexed by level if categorical) } }
        start = time.time()
        np_path = os.path.join(config.working_dir, 'np_data.npy')
        if not os.path.exists(np_path):
            print 'DATASET: parsing data into np arrays...'
            self.np_data = self._get_np_data()
            np.save(np_path, self.np_data)
        else:
            print 'DATASET: restoring np_arrays from ', np_path
            self.np_data = np.load(np_path)[()]
        print '\tdone, took %.2fs' % (time.time() - start)


    def datafile_to_np(self, datafile, feature_id_map=None, text_file=False):
        """ returns an np array of a 1-per-line datafile
            if feature_id_map is provided, the variable is assumed
                to be categorical, and the returned array will
                have one-hot rows whose ids correspond to the 
                values of the provided feature_id_map
        """
        num_examples = utils.file_len(datafile)
        num_features = len(feature_id_map) if feature_id_map else 1
        out = np.zeros((num_examples, num_features))

        for i, line in enumerate(open(datafile)):
            line = line.strip()
            if text_file:
                # text
                for feature in line.split()[:self.config.max_seq_len]:
                    out[i][feature_id_map.get(feature, UNK_ID)] += 1
            elif feature_id_map is not None:
                if line == '':
                    line = self.config.unk
                # categorical
                out[i][feature_id_map.get(line.replace(' ', '_'), UNK_ID)] += 1
            else:
                if line == '':
                    line = 0
                # continuous
                out[i][0] = float(line)
        return sparse.csr_matrix(out)


    def _get_np_data(self):
        """ TODO -- switch to 
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        """
        np_data = defaultdict(dict)
        for split, variables in self.data_files.items():
            for varname, filepath in variables.items():
                var = self.get_variable(varname)
                if not var['type'] == 'text' and var['skip']:
                    continue
                if var['type'] == 'continuous':
                    np_data[split][varname] = self.datafile_to_np(
                        datafile=filepath)
                else:
                    if varname == self.input_varname():
                        np_data[split][varname] = self.datafile_to_np(
                            datafile=filepath,
                            feature_id_map=self.features,
                            text_file=True)
                    else:
                        np_data[split][varname] = self.datafile_to_np(
                            datafile=filepath,
                            feature_id_map=self.class_to_id_map[varname])
        return np_data



    def _get_categorical_tables(self):
        """ generate ids for each categorical variable
        """
        class_to_id_map = defaultdict(dict)
        id_to_class_map = defaultdict(dict)
        for variable in self.config.data_spec[1:]:
            if variable['type'] != "categorical" or variable['skip']: 
                continue
            i = 0
            var_filename = self.whole_data_files[variable['name']]
            for level in set(open(var_filename).read().split('\n')):  # unique rows
                level = level.strip()
                if level == '':
                    level = self.config.unk
                if level in class_to_id_map[variable['name']]: 
                    continue
                level = level.replace(' ', '_') # whitespaces not allowed in class names
                class_to_id_map[variable['name']][level] = i
                id_to_class_map[variable['name']][i] = level
                i += 1
        return class_to_id_map, id_to_class_map



    def y_chunk(self, target_name, target_level=None, start=0, end=None):
        end = min(end, self.split_sizes[self.split]) if end else self.split_sizes[self.split]
        # pull out the target for the chunk
        y = self.np_data[self.split][target_name][start:end].toarray()
        if target_level is not None:
            target_col = self.class_to_id_map[target_name][target_level]
            y = y[:, target_col]
        y = np.squeeze(y)
        return y


    def text_X_chunk(self, feature_subset=None, start=0, end=None):
        end = min(end, self.split_sizes[self.split]) if end else self.split_sizes[self.split]
        # start with all the text features for desired chunk
        X = self.np_data[self.split][self.input_varname()][start:end].toarray()
        X_features = self.ordered_features
        if feature_subset is not None:
            assert len(feature_subset) > 0
            retain_indices = map(lambda f: self.features[f], feature_subset)
            X = X[:, retain_indices]
            X_features = feature_subset
        return X, X_features        


    def nontext_X_chunk(self, features, start=0, end=None):
        end = min(end, self.split_sizes[self.split]) if end else self.split_sizes[self.split]

        X_features = []
        cols = []
        for varname in features:
            one_hots = self.np_data[self.split][varname][start:end].toarray()
            for level, idx in self.class_to_id_map[varname].items():
                cols.append(one_hots[:, idx])
                X_features.append('%s|%s' % (varname, level))

        X = np.column_stack(cols)
        return X, X_features


    def input_varname(self):
        return self.config.data_spec[0]['name']

    def get_variable(self, varname):
        return next((v for v in self.config.data_spec if v['name'] == varname))


    def set_active_split(self, split):
        """ points the dataset towards a split
        """
        self.split = split


    def num_examples(self):
        """ number of batches in current split
        """
        examples = sum(1 for _ in open(self.data_files[self.split][self.input_varname()]))
        return examples

    def get_tokenized_input(self):
        return [
            line.strip().split() \
            for line in open(self.data_files[self.split][self.input_varname()])
        ]


    def data_for_var(self, var):
        """ TODO -- REFACTOR AWAY SINCE REDUNDENT WITH SELF.NP_DATA STUFF"""
        eval_fn = str if var['type'] == 'categorical' else float
        return [
            eval_fn(x.strip().replace(' ', '_')) \
            for x in open(self.data_files[self.split][var['name']])
        ]


    def num_levels(self, name):
        """ num levels for some categorical var
        """
        return len(self.class_to_id_map[name])


    def num_classes(self, varname):
        return len(self.class_to_id_map[varname])


    def _check_vocab(self, vocab_file):
        assert os.path.exists(vocab_file), "The vocab file %s does not exist" % vocab_file

        lines = map(lambda x: x.strip(), open(vocab_file).readlines())
    
        if len(lines) != len(set(lines)):
            print 'DATASET: vocab %s contains dups!! fixing.' % vocab_file
            unk = self.config.unk
            os.system('rm %s' % vocab_file)
            s = unk + '\n' + '\n'.join([x for x in set(lines) if x != unk])
            with open(vocab_file, 'w') as f: f.write(s)
            lines = map(lambda x: x.strip(), open(vocab_file).readlines())

        assert lines[0] == self.config.unk, \
            "The first words in %s is not %s" % (vocab_file)

        return len(lines)


    def _gen_vocab(self, text_file):
        vocab_path = os.path.join(self.base_dir, 'freq_vocab.txt')
        if not os.path.exists(vocab_path):
            print 'DATASET: generating vocab of %d tokens..' % self.config.vocab['top_n']
            start = time.time()
            word_ctr = Counter(open(text_file).read().split())
            vocab = map(lambda x: x[0], word_ctr.most_common(self.config.vocab['top_n']))
            with open(vocab_path, 'w') as f:
                f.write('\n'.join(vocab))
            print '\tdone. took %.fs' % (time.time() - start)
        else:
            print 'DATASET: restoring vocab from ', vocab_path
            vocab = [x.strip() for x in open(vocab_path).readlines()][1:] # unk is 0th elem

        if self.config.vocab['preselection_algo'] == 'identity':
            out_path = vocab_path

        elif self.config.vocab['preselection_algo'] == 'odds-ratio':
            out_path = os.path.join(self.base_dir, 'or_vocab.txt')
            if not os.path.exists(out_path):
                start = time.time()
                print 'ODDS_RATIO: selecting initial featureset'
                vocab = odds_ratio.select_features(
                    dataset=self, 
                    vocab=vocab, 
                    k=self.config.vocab['preselection_features'])
                print '\n\tdone. Took %.2fs' % (time.time() - start)
            else:
                print 'ODDS_RATIO: recoveing from ', out_path
                vocab = [x.strip() for x in open(out_path).readlines()]

        elif self.config.vocab['preselection_algo'] == 'mutual-information':
            out_path = os.path.join(self.base_dir, 'mi_vocab.txt')
            if not os.path.exists(out_path):
                start = time.time()
                print 'MUTUAL_INFORMATION: selecting initial featureset'
                vocab = mutual_information.select_features(
                    dataset=self, 
                    vocab=vocab, 
                    k=self.config.vocab['preselection_features'])
                print 'MUTUAL_INFORMATION: recoveing from ', out_path
                print '\n\tdone. Took %.2fs' % (time.time() - start)
            else:
                vocab = [x.strip() for x in open(out_path).readlines()]

        vocab = [self.config.unk] + vocab

        with open(out_path, 'w') as f:
            f.write('\n'.join(vocab))
        return out_path


    def _cut_data(self):
        """ break a dataset tsv into one file per variable and return pointers
                to each file
        """
        c = self.config

        split_sizes = {}

        data_prefix = os.path.join(c.data_dir, c.prefix)
        variable_paths = defaultdict(dict)
        whole_data_paths = {}
        for split_suffix in [c.train_suffix, c.dev_suffix, c.test_suffix]:
            file = data_prefix + split_suffix
            assert os.path.exists(file), 'Split %s doesnt exist' % file

            split_sizes[split_suffix] = utils.file_len(file)

            for i, variable in enumerate(c.data_spec):
                if i > 0 and variable['skip']: 
                    continue
                variable_path = data_prefix + '.' + variable['name'] + split_suffix
                variable_path_nosplit = data_prefix + '.' + variable['name']

                variable_paths[split_suffix][variable['name']] = variable_path
                whole_data_paths[variable['name']] = variable_path_nosplit

                if not os.path.exists(variable_path):
                    os.system('cat %s | cut -f%d > %s' % (
                        file, i+1, variable_path))
                if not os.path.exists(variable_path_nosplit):
                    os.system('cat %s | cut -f%d > %s' % (
                        data_prefix, i+1, variable_path_nosplit))

        return variable_paths, split_sizes, whole_data_paths


    def cleanup(self):
        """ cleanup all the per-variable files created by _cut_data
        """
        for _, splits in self.data_files.iteritems():
            for _, filepath in splits.iteritems():
                os.system('rm %s' % filepath)



    def make_tf_iterators(self, params):
        """
            returns a dictionary mapping each variable
                to a tf iterator's placeholder, along with
                the special key 'initializer' which maps to 
                the initializer for this iterator
        """

        vocab_table = lookup_ops.index_table_from_file(
            self.vocab, default_value=UNK_ID)
        eos_id = tf.cast(
            vocab_table.lookup(tf.constant(self.config.eos)),
            tf.int32)

        def text_dataset(file):
            dataset = tf.contrib.data.TextLineDataset(file)
            # break sentences into tokens
            dataset = dataset.map(lambda txt: tf.string_split([txt]).values)
            # convert to ids
            dataset = dataset.map(lambda txt: (
                txt, tf.cast(vocab_table.lookup(txt), tf.int32)))

            # now cut off
            maxlen = self.config.max_seq_len
            dataset = dataset.map(lambda txt, ids: (txt[:maxlen], ids[:maxlen]))

            # add lengths
            dataset = dataset.map(lambda txt, ids: (txt, ids, tf.size(ids)))


            return dataset

        def continuous_dataset(file):
            def is_number(x):
                try:
                    float(x)
                    return True
                except:
                    return False

            dataset = tf.contrib.data.TextLineDataset(file)
            # apend 0 to start in case there's blank rows
            dataset = dataset.map(
                lambda x: tf.string_to_number(x) \
                            if is_number(x) else tf.string_to_number('0'))
            return dataset

        def categorical_dataset(file, variable_name):
            dataset = tf.contrib.data.TextLineDataset(file)

            classes_ids = self.class_to_id_map[variable_name]
            # include spaces because they're in raw mapping, but not
            #    self.class_to_id_map
            dict_with_spaces = {level.replace("_",' '): idx for level, idx in classes_ids.items()}
            dict_with_spaces.update(classes_ids)
            class_lookup_table = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(
                    keys=dict_with_spaces.keys(),
                    values=dict_with_spaces.values(),
                    key_dtype=tf.string,
                    value_dtype=tf.int32), UNK_ID)
            dataset = dataset.map(lambda x: class_lookup_table.lookup(x))
            return dataset


        def batch_up(datset):
            # first element is (text, text, text len), followed by all other vars
            num_variables = len([v for v in self.config.data_spec[1:] if not v['skip']])
            padded_shapes = tuple(
                [(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]))] + [
                tf.TensorShape([]) for _ in range(num_variables)])

            # pad text with eos, otherwise 0 (means unused)
            padding_values = [(self.config.eos, eos_id, 0)]
            # but hast to be 0.0 for tf.float32 (aka scalars) and 0 for tf.int32
            # (aka categorical)
            for var in self.config.data_spec[1:]:
                if var['skip']:
                    continue

                if var['type'] == 'categorical':
                    padding_values.append(0)
                else:
                    padding_values.append(0.0)
            padding_values = tuple(padding_values)

            return datset.padded_batch(
                params['batch_size'],
                padded_shapes=padded_shapes,
                padding_values=padding_values)

        datasets = []
        for i, variable in enumerate(self.config.data_spec):
            if i > 0 and variable['skip']:
                continue

            data_file = self.data_files[self.split][variable['name']]
            if variable['type'] == 'text':
                dataset = text_dataset(data_file)
            elif variable['type'] == 'continuous':
                dataset = continuous_dataset(data_file)
            else:
                dataset = categorical_dataset(data_file, variable['name'])
            datasets.append(dataset)

        dataset = tf.contrib.data.Dataset.zip(tuple(datasets))
        dataset = batch_up(dataset)

        out = {}
        iterator = dataset.make_initializable_iterator()
        data_spec = [v for v in self.config.data_spec if not v.get('skip', False)]
        placeholders = iterator.get_next()        
        for i, (placeholder, variable) in enumerate(zip(placeholders, data_spec)):
            out[variable['name']] = placeholder
        out['initializer'] = iterator.initializer
        return out
















