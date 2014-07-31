#!/usr/bin/env python
# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Build commands and dependencies from JSON configuration files.
"""
import os
import json
import warnings

import sklearn.externals.joblib as joblib

from genericML.crossval import make_folds
from genericML.configuration import check_conf
from genericML.workflow import create_wf, save_wf
from genericML.dynamic import load_data, get_grid
from genericML.arguments import get_cmd_builder_argparser

verbose = False

if __name__ == "__main__": 
    # parse command line arguments
    args = get_cmd_builder_argparser().parse_args()
    if args.no_warn:
        warnings.simplefilter("ignore")
    else:
        warnings.simplefilter("default")
    verbose = args.verbose
    if verbose:
        print("=======")
        print(args)
        print("=======")

    # step 1: read data file
    with open(args.data, 'r') as fd:
            data_cfg = dict(json.load(fd), src=args.data)
    check_conf(data_cfg, cat="data", verbose=verbose)
    X, Y = load_data(data_cfg)
    n_samples = X.shape[0]
    n_targets = 1 if (Y.ndim == 1) else Y.shape[1]
    if n_targets > 1:
        warnings.warn("More than one target. Unexpected results or crashes "
                      "may occur if your methods and/or metrics "
                      "cannot handle multiple target.", RuntimeWarning)

    # step 2: read crossval file
    with open(args.crossval, 'r') as fd:
            cv_cfg = dict(json.load(fd), 
                          src=os.path.basename(args.crossval))
    check_conf(cv_cfg, cat="crossval", verbose=verbose)

    # step 3: load method configuration
    with open(args.method, 'r') as fd:
        method_cfg = dict(json.load(fd), 
                          src=os.path.basename(args.method))
    check_conf(method_cfg, cat="method", verbose=verbose)

    # step 4: write dataset file(s)
    # generate folds/grid and write dataset
    output_file = os.path.join(args.outputdir, "dataset.joblib")
    folds = dict(make_folds(Y, cv_cfg, verbose=verbose), 
                 src=os.path.basename(output_file))
    if cv_cfg["modelSelection"]:
        grid = get_grid(cv_cfg, X, Y)
    else:
        grid = np.asarray([method_cfg[method_cfg["est_param"]]])
    if verbose:
        print("Input dataset destination: {}".format(output_file))
    dataset = {"X": X, "Y": Y, 
               "n_samples": n_samples, "n_targets": n_targets,
               "folds": folds, "grid": grid}
    joblib.dump(dataset, output_file, compress=1)

    # step 5: construct commands or workflow
    wf = create_wf(folds, cv_cfg, data_cfg, method_cfg, verbose=verbose)
    wf_out = os.path.join(args.outputdir, "workflow.json")
    save_wf(wf, wf_out, mode=args.output_mode)
    if verbose:
        print("Write workflow to: {}".format(wf_out))
