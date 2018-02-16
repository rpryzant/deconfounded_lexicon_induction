""" correlation calculation """
import sys
import numpy as np
from tqdm import tqdm

np.warnings.filterwarnings('ignore')


def cramers_v(feature, text, targets, possible_labels):
    """ feature: string
        text: list( list(words) )
        targets: list(string)
        possible_labels: list(string)

        chisq statistic for a single feature, given some text
        and target info (Y) and possible_labels (possible values for Y)
    """
    obs = np.zeros( (2, len(possible_labels)) )
    for description, target in zip(text, targets):
        if feature in description:
            obs[1, possible_labels.index(target)] += 1
        else:
            obs[0, possible_labels.index(target)] += 1

    row_totals = np.sum(obs, axis=1)
    col_totals = np.sum(obs, axis=0)
    n = np.sum(obs)
    expected = np.outer(row_totals, col_totals) / n
    chisq = np.sum( np.nan_to_num(((obs - expected) ** 2 ) / expected ))

    r = 2
    k = len(possible_labels)
    phisq = chisq / n
    V = np.sqrt(phisq / min(k-1, r-1))
    return V


def pointwise_biserial(feature, text, targets):
    """ feature: string
        text: list( list(words) )
        targets: list(float)

    pointwise biserial statistic
    https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient
    """
    s = np.std(targets)

    group0 = []
    group1 = []
    for text_example, val in zip(text, targets):
        if val == -1:
            continue
        if feature in text_example:
            group0.append(val)
        else:
            group1.append(val)

    m0 = np.mean(group0)
    m1 = np.mean(group1)

    n0 = float(len(group0))
    n1 = float(len(group1))
    n = n0 + n1

    rpb = (abs(m1 - m0) / s) * np.sqrt((n0 * n1) / (n ** 2))
    if type(rpb) == type(0.0):
        print 'here'
        return None
    return rpb

