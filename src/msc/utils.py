import cPickle
import bisect
import time
import tensorflow as tf
import sys
import numpy as np
import yaml
from collections import namedtuple
from collections import OrderedDict

class UnsortableList(list):
    def sort(self, *args, **kwargs):
        pass

class UnsortableOrderedDict(OrderedDict):
    def items(self, *args, **kwargs):
        return UnsortableList(OrderedDict.items(self, *args, **kwargs))

yaml.add_representer(UnsortableOrderedDict, yaml.representer.SafeRepresenter.represent_dict)


def standardize(arr):
    """ standardize a np array arr """
    return (arr - np.mean(arr)) / np.std(arr)

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False


def swap(a, i, j):
    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp


def pickle(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)

def depickle(path):
    with open(path, 'r') as f:
        return cPickle.load(f)

def percentile(x, threshold):
    # return number at threshold^th percentile of x
    idx = int(threshold * len(x))
    return sorted(x)[idx]


def rank_threshold(x, rank, largest=True):
    # return the threshold for the "rank" largest numbers in x
    if largest:
        return sorted(x)[::-1][rank]
    else:
        return sorted(x)[rank]


def nested_iter(obj):
    """ recursively iterate through nested dict
    """
    for k, v in d.iteritems():
        if not isinstance(d, dict):
            yield k, v
        else:
            for k2, v2 in nested_iter(v):
                yield k2, v2

def file_len(fp):
    return sum(1 for _ in open(fp))


def add_summary(summary_writer, global_step, name, value):
  """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., name=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
  summary_writer.add_summary(summary, global_step)



def load_config(filename):
    """ loads YAML config into a named tuple
    """
    d = yaml.load(open(filename).read())
    d = namedtuple("config", d.keys())(**d)
    return d


def write_config(config, path):
    """ writes a config object to path
        TODO -- write in same order???
    """
    yaml_str = yaml.dump(dict(config._asdict()), default_flow_style=False)
    with open(path, 'w') as f:
        f.write(yaml_str)


class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)
