#!/usr/bin/env python
# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Generic mapper
"""
import json
import numpy as np

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from mempamal.arguments import get_map_argparser
from mempamal.crossval import get_fold, print_fold
from mempamal.generic_gridsearch import GenericGridSearch
from mempamal.dynamic import construct_pipeline, get_score_func

verbose = False

if __name__ == "__main__":
    # parse command line arguments
    args = get_map_argparser().parse_args()
    verbose = args.verbose
    if verbose:
        print("=======")
        print(args)
        print("=======")

    # read data and configuration files
    dataset = joblib.load(args.dataset)
    grid = dataset["grid"]
    with open(args.crossval, 'r') as fd:
        cv_cfg = json.load(fd)
    with open(args.method, 'r') as fd:
        method_cfg = json.load(fd)

    # construct folds
    train_index, test_index = get_fold(dataset["folds"], 
                                       args.outer, inner=args.inner)
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
    which_cv = ("gridSearch" if cv_cfg["modelSelection"] 
                else "crossval_score")
    score_func, score_kwargs = get_score_func(cv_cfg, cv=which_cv)
    clf = GenericGridSearch(est=Pipeline,
                            est_param=est_param,
                            params=grid,
                            est_kwargs=est_kwargs,
                            score_func=score_func,
                            score_kwargs=score_kwargs)

    # fit/predict/score
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)    
    if verbose:
        print Y_test
        print Y_pred
    scores = clf.score(Y_test, Y_pred)
    res = ({"scores": scores} if cv_cfg["modelSelection"] 
           else {"scores": scores[0]})

    # save result
    joblib.dump(res, args.out, compress=1)
