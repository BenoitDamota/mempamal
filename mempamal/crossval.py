# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Functions relative to cross-validation and folds iterator.
"""
import numpy as np

from .dynamic import dynamic_import


def _construct_folds_iterator(y, cfg, key):
    """Dynamically construct a fold iterator from the configuration (cfg, key).

    Instantiate a folds iterator depending on the cross-validation
    configuration. If the folds iterator is statified, it should take the
    targets array y, else it should take only ne number of samples.

    Notes
    -----
    See sklearn.cross_validation for more information about how a
    folds iterator should work

    Parameters
    ----------
    y : array, shape (n_samples, n_targets)
        targets array
    cfg : dict,
        configuration dict for cross-validation.
    key : str,
        key in cfg that provide a key "foldsIterator" and a list
        [class, dict] where 'class' is a string representing the
        folds iterator object to import and 'dict' are the keywords
        arguments.
    """
    cv_cfg = cfg[key]
    dyn_cla = dynamic_import(cv_cfg["foldsIterator"][0])
    kwargs = cv_cfg["foldsIterator"][1]
    if cfg["stratified"]:
        fg = dyn_cla(y, **kwargs)
    else:
        fg = dyn_cla(y.shape[0], **kwargs)
    return fg


def make_folds(y, cfg, verbose=False):
    """Constructs the folds from crossval configuration

    Parameters
    ----------
    y : array, shape (n_samples, n_targets)
        targets array
    cfg : dict,
        configuration dict for cross-validation.
    verbose : boolean, optional
        verbose mode
    """
    folds = []
    keys = []
    if cfg["stratified"] and y.ndim > 1:
        raise TypeError("Stratified and multi-target are incompatible.")
    else:
        # construct the outer cross-validation folds iterator
        fg = _construct_folds_iterator(y, cfg, "crossval_score")
        for i, (train, test) in enumerate(fg):
            if verbose:
                print("Outer Fold {}:\n{}\n{}\n---".format(
                        i, np.sort(y[train]), np.sort(y[test])))
            folds.append((train, test))
            keys.append("%d" % i)
            if cfg["modelSelection"]:
                yi = y[train]
                # construct the inner cross-validation folds iterator
                fgi = _construct_folds_iterator(yi, cfg, "gridSearch")
                for k, (itrain, itest) in enumerate(fgi):
                    if verbose:
                        print("Inner Fold {}:\n{}\n{}\n---".format(
                                k, np.sort(yi[itrain]), np.sort(yi[itest])))
                    folds.append((itrain, itest))
                    keys.append("%d_%d" % (i, k))
    # associate string identifiers with corresponding folds
    if cfg["modelSelection"]:
        folds_dic = dict(zip(keys, folds), n_inner=(k + 1), n_outer=(i + 1))
    else:
        folds_dic = dict(zip(keys, folds), n_outer=(i + 1))
    return folds_dic


def get_fold(folds, outer, inner=None):
    """Get a given folds from the folds dict (see make_folds)

    Parameters
    ----------
    folds : dict,
        dictionnary with all the folds.
    outer : int,
        ID of the outer fold.
    inner : int, optional (default=None)
        ID of the inner fold.
    """
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
    """Pretty printer for a couple of train/test folds.

    """
    print(train_index)
    print(test_index)
    print("train/test sizes: {}/{}".format(
            train_index.size,
            test_index.size))
    print("=======")
