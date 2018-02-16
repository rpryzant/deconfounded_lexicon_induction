import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import src.msc.utils as utils
import math
import os
import time
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np


def compute_mi(dataset, target_name, vocab, level=None):
    """ compute the mutual information for each token in a lexicon
    """
    if not level:
        # bucket based on bottom/top 30%
        response = dataset.datafile_to_np(
            datafile=dataset.whole_data_files[target_name])
        response = response.toarray()
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
    feature_counts = defaultdict(lambda: {
        'n00': 1.,  # docs without term, 0 label
        'n01': 1.,  # docs without term, 1 label
        'n10': 1.,  # docs with term, 0 label
        'n11': 1.   # docs with term, 1 label
    })

    input_text = open(dataset.whole_data_files[dataset.input_varname()])

    for line, label in zip(input_text, response):
        if not label in [0, 1]: continue
        line = set(line.strip().split())
        for feature in vocab:
            if label == 0:
                if feature in line:
                    feature_counts[feature]['n10'] += 1
                else:
                    feature_counts[feature]['n00'] += 1
            else:
                if feature in line:
                    feature_counts[feature]['n11'] += 1
                else:
                    feature_counts[feature]['n01'] += 1

    def mi(n00, n01, n10, n11):
        n0_ = n00 + n01   # docs without term
        n1_ = n11 + n10   # docs with term
        n_0 = n10 + n00   # docs with 0 label
        n_1 = n11 + n01   # docs with 1 label
        n = n00 + n01 + n11 + n10   # total n    

        mutual_info = (n11/n) * math.log((n * n11) / (n1_ * n_1)) + \
                      (n01/n) * math.log((n * n01) / (n0_ * n_1)) + \
                      (n10/n) * math.log((n * n10) / (n1_ * n_0)) + \
                      (n00/n) * math.log((n * n00) / (n0_ * n_0))        
        return mutual_info

    MIs = dict(map(lambda (f, d): (f, mi(**d)), feature_counts.items()))
    return MIs

def select_features(dataset, vocab, k):
    """ use mutual information to select features """
    confounds = [v for v in dataset.config.data_spec[1:] \
                if v['control'] and not v['skip']]

    # {feature: [odds ratio for each confound]}
    feature_ratios = defaultdict(list)
    for var in tqdm(confounds):
        if var['type'] == 'categorical':
            for level in tqdm(dataset.class_to_id_map[var['name']]):
                MIs = compute_mi(
                    dataset=dataset,
                    target_name=var['name'],
                    level=level,
                    vocab=vocab)
        else:
            MIs = compute_mi(
                dataset=dataset,
                target_name=var['name'],
                vocab=vocab)
        for f, x in MIs.items():
            feature_ratios[f].append(x)

    feature_importance = sorted(
        map(lambda (f, x): (np.mean(x), f), feature_ratios.items()))

    # write this to output
    with open(os.path.join(dataset.config.working_dir, 'mutual-information-scores-before-selection.txt'), 'w') as f:
        s = '\n'.join('%s\t%s' % (f, str(x)) for x, f in feature_importance)
        f.write(s)

    # choose K features with smallest MI
    selected_features = feature_importance[:k]
    selected_features = map(lambda (x, f): f, selected_features)
    return selected_features



