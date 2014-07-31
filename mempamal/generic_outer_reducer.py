#!/usr/bin/env python
# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
""" 
Generic outer reducer.
"""
from glob import glob

import numpy as np
from sklearn.externals import joblib

from genericML.arguments import get_ored_argparser

verbose = False

if __name__ == "__main__":
    # parse command line arguments
    args = get_ored_argparser().parse_args()
    verbose = args.verbose
    if verbose:
        print("=======")
        print(args)
        print("=======")

    # retrieve results from (outer) folds  
    file_pattern = (args.__getattribute__("in")).format(outer="*")
    list_files = glob(file_pattern)
    scores = []
    for cur_file in list_files:
         if verbose:
             print("Reading {}".format(cur_file))
         cur_ar = joblib.load(cur_file)
         scores.append(cur_ar["scores"])
    raw =  np.asarray(scores)

    # summary
    res = {"raw": raw, 
           "mean": np.mean(raw, axis=0),
           "median": np.median(raw, axis=0),
           "std": np.std(raw, axis=0)}
    if verbose:
        print("=======")
        print(res)
        print("=======")
    print("Cross-validated score(s):")
    print("  Mean  : %s" % (res['mean']).__str__())
    print("  Median: %s" % (res['median']).__str__())
    print("  Std   : %s" % (res['std']).__str__())

    # save result
    joblib.dump(res, args.out, compress=1)
