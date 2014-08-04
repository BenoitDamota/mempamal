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

from mempamal.configuration import check_conf, build_dataset
from mempamal.workflow import create_wf, save_wf
from mempamal.dynamic import load_data
from mempamal.arguments import get_cmd_builder_argparser

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
    in_out_dir = data_cfg["in_out_dir"]

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
    dataset = build_dataset(X, Y, method_cfg, cv_cfg, args.outputdir,
                            verbose=verbose)

    # step 5: construct commands or workflow
    wf = create_wf(dataset['folds'], cv_cfg, method_cfg, in_out_dir,
                   verbose=verbose)
    wf_out = os.path.join(args.outputdir, "workflow.json")
    save_wf(wf, wf_out, mode=args.output_mode)
    if verbose:
        print("Write workflow to: {}".format(wf_out))
