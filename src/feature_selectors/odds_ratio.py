import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import src.msc.utils as utils
import time
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os

def compute_ratios(dataset, target_name, vocab, level=None):
    """ computes odds ratios, returning a dict of each feature's ratio
        uses a subset of features, defined by feature_indices
    """
    if not level:
        # bucket based on bottom/top 30%
        response = dataset.datafile_to_np(
            datafile=dataset.whole_data_files[target_name])
        low_threshold = utils.percentile(response, 0.3)
        high_threshold = utils.percentile(response, 0.7)
        response[response < low_threshold] = 0
        response[response > high_threshold] = 1
    else:
        level_idx = dataset.class_to_id_map[target_name][level]
        response = dataset.datafile_to_np(
            datafile=dataset.whole_data_files[target_name],
            feature_id_map=dataset.class_to_id_map[target_name])
        response = np.squeeze(response[:, dataset.class_to_id_map[target_name][level]].toarray())

    vocab = set(vocab)
    feature_counts = defaultdict(lambda: {0: 0, 1: 0})

    input_text = open(dataset.whole_data_files[dataset.input_varname()])

    for line, label in zip(input_text, response):
        if not label in [0, 1]: continue
        line = line.strip().split()
        for feature in line:
            # ignore if not feature of interest or outside of bucketing
            if not feature in vocab:  continue

            feature_counts[feature][label] += 1

    ratios = {}
    for feature, counts in feature_counts.iteritems():
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2938757/
        a = feature_counts[feature][0]
        b = feature_counts[feature][1]
        c = len(response[response == 0]) - a
        d = len(response[response == 1]) - b
        try:
            ratios[feature] = float(a * d) / (b * c)
        except ZeroDivisionError:
            pass

    return ratios


def select_features(dataset, vocab, k):
    """ use odds ratio to select features
        not joined with MI because code paths differ significantly
    """
    confounds = [v for v in dataset.config.data_spec[1:] \
                if v['control'] and not v['skip']]

    # {feature: [odds ratio for each confound]}
    feature_ratios = defaultdict(list)
    for var in tqdm(confounds):
        if var['type'] == 'categorical':
            for level in tqdm(dataset.class_to_id_map[var['name']]):
                ratios = compute_ratios(
                    dataset=dataset,
                    target_name=var['name'],
                    level=level,
                    vocab=vocab)
        else:
            ratios = compute_ratios(
                dataset=dataset,
                target_name=var['name'],
                vocab=vocab)
        for f, x in ratios.items():
            feature_ratios[f].append(x)

    feature_importance = sorted(
        map(lambda (f, x): (np.mean(x), f), feature_ratios.items()))

    # write this to output
    with open(os.path.join(dataset.config.working_dir, 'odds-ratio-scores-before-selection.txt'), 'w') as f:
        s = '\n'.join('%s\t%s' % (f, str(x)) for x, f in feature_importance)
        f.write(s)

    # choose K features with smallest odds ratio
    selected_features = feature_importance[:k]
    selected_features = map(lambda (x, f): f, selected_features)
    return selected_features
