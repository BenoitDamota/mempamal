#!/usr/bin/env python
# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Generic inner reducer
"""
import json
import numpy as np

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from mempamal.arguments import get_ired_argparser
from mempamal.crossval import get_fold, print_fold
from mempamal.gridsearch import GenericGridSearch
from mempamal.dynamic import construct_pipeline, get_score_func

verbose = False

if __name__ == "__main__":
    # parse command line arguments
    args = get_ired_argparser().parse_args()
    verbose = args.verbose
    if verbose:
        print("=======")
        print(args)
        print("=======")

    # read files
    dataset = joblib.load(args.dataset)
    with open(args.method, 'r') as fd:
        method_cfg = json.load(fd)
    with open(args.crossval, 'r') as fd:
        cv_cfg = json.load(fd)

    # retrieve results from inner folds
    n_inner = dataset["folds"]["n_inner"]
    n_targets = dataset["n_targets"]
    grid = dataset["grid"]
    scores = np.zeros((n_inner, n_targets, len(grid)))
    for i in range(n_inner):
        cur_file = (args.__getattribute__("in")).format(inner=i)
        if verbose:
            print("Reading {}".format(cur_file))
        cur_ar = joblib.load(cur_file)
        scores[i] = cur_ar["scores"]
    if verbose:
        print("=======")
    # Parameter selection:
    # mean on the folds and target, i.e. select the best parameters
    # independently of the target (that's one possible strategy for
    # multiple targets)
    ms = np.mean(scores.reshape((-1, len(grid))), axis=0)
    bid = np.where(ms == np.amax(ms))[0]
    best_param = grid[bid[0]]

    # construct folds
    train_index, test_index = get_fold(dataset["folds"], args.outer)
    if verbose:
        print_fold(train_index, test_index)
    X = dataset["X"]
    Y = dataset["Y"]
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]

    # construct estimator
    est_kwargs, est_param = construct_pipeline(method_cfg)
    score_func, score_kwargs = get_score_func(cv_cfg, cv="gridSearch")
    clf = GenericGridSearch(est=Pipeline,
                            params=[best_param],
                            est_kwargs=est_kwargs,
                            score_func=score_func,
                            score_kwargs=score_kwargs)

    # fit/predict/score
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    if verbose:
        print Y_test
        print Y_pred[0]
    res = {"scores": clf.score(Y_test, Y_pred)[0]}
    print("Best parameters set: {}".format(best_param))
    print("scores: {}".format(res["scores"]))

    # save result
    joblib.dump(res, args.out, compress=1)
