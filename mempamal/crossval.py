# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Functions relative to cross-validation and folds iterator.
"""
import numpy as np

def _construct_folds_iterator(y, cfg, key):
    cv_cfg = cfg[key]
    it = cv_cfg["foldsIterator"][0]
    kwargs = cv_cfg["foldsIterator"][1]
    mod, cla = it.rsplit('.', 1)
    dyn_cla = getattr(__import__(mod, fromlist=[str(cla)]), cla)
    if cfg["stratified"]:
        fg = dyn_cla(y, **kwargs)
    else:
        fg = dyn_cla(y.shape[0], **kwargs)
    return fg

def make_folds(y, cfg, verbose=False):
    """
    Constructs folds from crossval configuration
    """
    folds = []
    keys = []
    if cfg["stratified"] and y.ndim > 1:
        raise TypeError("Stratified and multi-target are incompatible.")
    else:
        fg = _construct_folds_iterator(y, cfg, "crossval_score")
        for i, (train, test) in enumerate(fg):
            if verbose:
                print("Outer Fold {}:\n{}\n{}\n---".format(
                        i, np.sort(y[train]), np.sort(y[test])))
            folds.append((train, test))
            keys.append("%d" % i)
            if cfg["modelSelection"]:
                yi = y[train]
                fgi = _construct_folds_iterator(yi, cfg, "gridSearch")
                for k, (itrain, itest) in enumerate(fgi):
                    if verbose:
                        print("Inner Fold {}:\n{}\n{}\n---".format(
                                k, np.sort(yi[itrain]), np.sort(yi[itest])))
                    folds.append((itrain, itest))
                    keys.append("%d_%d" % (i, k))
    if cfg["modelSelection"]:
        folds_dic = dict(zip(keys, folds), n_inner=(k + 1), n_outer=(i + 1))
    else:
        folds_dic = dict(zip(keys, folds), n_outer=(i + 1))
    return folds_dic

def get_fold(folds, outer, inner=None):
    try:
        if inner is None:
            return folds["%d" % outer]
        else:
            outer_ind = folds["%d" % outer][0]
            inner_ind = folds["%d_%d" % (outer, inner)]
            return outer_ind[inner_ind[0]], outer_ind[inner_ind[1]]
    except KeyError:
        raise KeyError("unexpected fold: outer={}, inner={}".format(
                outer, inner))

def print_fold(train_index, test_index):
    print train_index
    print test_index
    print("train/test sizes: %d/%d" % (
            train_index.size,
            test_index.size))
    print("=======")
