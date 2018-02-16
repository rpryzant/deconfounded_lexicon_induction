""" main entrypoint """

import yaml
import os
import csv
import argparse
from collections import namedtuple
import pickle
import random
import numpy as np
import tensorflow as tf
import time
import sys
import copy
from collections import defaultdict
import pandas as pd
import traceback

from src.data.dataset import Dataset
import src.msc.constants as constants
import src.msc.utils as utils
import src.analysis.evaluator as evaluator
import src.models.neural.tf_dummy as tf_dummy
import src.models.neural.tf_flipper as tf_flipper

def process_command_line():
    """ returns a 1-tuple of cli args
    """
    parser = argparse.ArgumentParser(description='usage')
    parser.add_argument('--config', dest='config', type=str, default='config.yaml', 
                        help='config file for this experiment')
    parser.add_argument('--test', dest='test', action='store_true', 
                        help='run test')
    parser.add_argument('--train', dest='train', action='store_true', 
                        help='run training')
    parser.add_argument('--gpu', dest='gpu', type=str, default='0', help='gpu')
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)  # only default graph


def run_experiment(config, args):
    # if train, switch the dataset to train, then
    #  train and save each model in the config spec
    set_seed(config.seed)

    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)
    utils.write_config(config, os.path.join(config.working_dir, 'config.yaml'))

    print 'MAIN: parsing dataset'
    d = Dataset(config, config.working_dir)
    print 'MAIN: dataset done. took %.2fs' % (time.time() - start)

    if args.train:
        d.set_active_split(config.train_suffix)

        for model_description in config.model_spec:
            if model_description.get('skip', False):
                continue

            print 'MAIN: training ', model_description['name']
            start_time = time.time()
            model_dir = os.path.join(config.working_dir, model_description['name'])
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model = constants.MODEL_CLASSES[model_description['type']](
                config=config, 
                params=model_description['params'])

            model.train(d, model_dir)
            model.save(model_dir)
            print 'MAIN: training %s done, time %.2fs' % (
                model_description['name'], time.time() - start_time)

    # if test, switch the datset to test, 
    #  and run inference + evaluation for each model
    #  in the config spec
    if args.test:
        d.set_active_split(config.test_suffix)
        results = []  # items to be written in executive summary 
        for model_description in config.model_spec:
            if model_description.get('skip', False):
                continue

            print 'MAIN: inference with ', model_description['name']
            start_time = time.time()

            model = constants.MODEL_CLASSES[model_description['type']](
                config=config, 
                params=model_description['params'])

            model_dir = os.path.join(config.working_dir, model_description['name'])
            model.load(d, model_dir)

            predictions = model.inference(d, model_dir) # scores, features_importance
            utils.pickle(
                predictions, os.path.join(model_dir, 'predictions'))

            evaluation = evaluator.evaluate(config, d, predictions, model_dir)
            utils.pickle(
                evaluation, os.path.join(model_dir, 'evaluation'))
            evaluator.write_summary(evaluation, model_dir)
            # store info for executive summary
            results.append({
                'model-name': model_description['name'],
                'model-type': model_description['type'],
                'params': str(model_description['params']),
                'correlation': evaluation['mu_corr'],
                'regression_performance': evaluation['mu_reg_perf'],
                'fixed_performance': evaluation['mu_fixed_perf'],
                'confound_performance': evaluation['mu_confound_perf'],
                'model_dir': model_dir,
            })

            print 'MAIN: evaluation %s done, time %.2fs' % (
                model_description['name'], time.time() - start_time)

        return results


def validate_data(config, args):
    """ make splits and clean out broken lines 
    """
    def validate_rowfile(in_path, out_path):
        out_file = open(out_path, 'w')

        for l in open(in_path):
            parts = l.strip().split('\t')
            # invalid number of cells
            if len(data_spec) != len(parts): 
                skipped_lines += 1
                continue
            skip = False
            for x, var in zip(parts, data_spec):
                if var.get('skip', False): continue

                if var['type'] == 'continuous' and not utils.is_number(x):
                    skip = True; break
                if x == '':
                    skip = True; break
            if skip:
                skipped_lines += 1
                continue
            out_file.write(l)
        out_file.close()


    skipped_lines = 0
    data_spec = config.data_spec

    d = copy.deepcopy(dict(config._asdict()))
    in_data_prefix = os.path.join(config.data_dir, config.prefix)
    out_data_dir = os.path.join(config.working_dir, 'data')
    out_data_prefix = os.path.join(out_data_dir, config.prefix)
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    print 'MAIN: making splits...'
    # makes fresh splits if not provided
    print '\t shuffling...'
    tmp_shuf_corpus = out_data_prefix
    validated_path = tmp_shuf_corpus + '.validated'
    if not os.path.exists(tmp_shuf_corpus):
        shuf = 'shuf' if 'linux' in sys.platform.lower() else 'gshuf'
        os.system('paste %s | %s > %s' % (in_data_prefix, shuf, tmp_shuf_corpus))
    validate_rowfile(tmp_shuf_corpus, validated_path)

    print '\t split:', config.train_suffix
    file_path = out_data_prefix + config.train_suffix
    validated_path = out_data_prefix + '.validated' + config.train_suffix
    if not os.path.exists(validated_path):
        os.system('tail -n +%d %s > %s' % (
            config.test_size + config.dev_size, tmp_shuf_corpus, file_path))
        validate_rowfile(file_path, validated_path)

    print '\t split:', config.dev_suffix
    file_path = out_data_prefix + config.dev_suffix
    validated_path = out_data_prefix + '.validated' + config.dev_suffix
    if not os.path.exists(validated_path):
        os.system('head -n %d %s > %s' % (
            config.dev_size, tmp_shuf_corpus, file_path))
        validate_rowfile(file_path, validated_path)

    print '\t split:', config.test_suffix
    file_path = out_data_prefix + config.test_suffix
    validated_path = out_data_prefix + '.validated' + config.test_suffix
    if not os.path.exists(validated_path):
        os.system('head -n %d %s | tail -n +%d > %s' % (
            config.dev_size + config.test_size, 
            tmp_shuf_corpus, 
            config.test_size,
            file_path))
        validate_rowfile(file_path, validated_path)

    d['data_dir'] = out_data_dir
    d['prefix'] = d['prefix'] + '.validated'

    return namedtuple("config", d.keys())(**d), skipped_lines



if __name__ == '__main__':
    # parse args
    args = process_command_line()
    config = utils.load_config(args.config)   

    # boilerplate
    reload(sys)
    sys.setdefaultencoding('utf8')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)

    # make splits data
    print 'MAIN: validating data...'
    start = time.time()
    config, skipped = validate_data(config, args)
    print '\t done. Took %.2fs, found %d invalid rows' % (
        time.time() - start, skipped)
 
    # boilerplate
    start = time.time()
    summary_path = os.path.join(config.working_dir, 'summary.csv')
    summary_file = open(summary_path, 'a')
    csv_writer = csv.writer(summary_file)

    # run expt
    results = run_experiment(config, args)
    if args.test:
        print 'MAIN: writing summary to ', summary_path
        for res in results:
            csv_writer.writerow(res.keys())
            csv_writer.writerow(res.values())

    summary_file.close()

