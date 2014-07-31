# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Functions relative to configuration file checking
"""

def _check_conf(cfg, req_keys, cat=""):
    for k in req_keys:
        try:
            cfg[k]
        except KeyError:
            raise KeyError(("Required key is missing in the "
                            "{} configuration file: {}").format(
                    cat, k))
    return

def _check_cv_conf(cfg):
    """
    Check the configuration of the cross-validation
    """
    req_keys = ["stratified", "modelSelection", "crossval_score"]
    _check_conf(cfg, req_keys, cat="crossval")
    req_keys = ["foldsIterator", "funcMetric"]
    _check_conf(cfg["crossval_score"], req_keys, cat="crossval")
    if cfg["modelSelection"]:
        req_keys = ["foldsIterator", "funcMetric", "parametersGrid"]
        _check_conf(cfg, ["gridSearch"], cat="crossval")
        _check_conf(cfg["gridSearch"], req_keys, cat="crossval")
    return

def _check_data_conf(cfg):
    """
    Check the configuration of the data
    """
    req_keys = ["func", "kwargs", "in_out_dir"]
    _check_conf(cfg, req_keys, cat="data")
    return

def _check_method_conf(cfg):
    """
    Check the configuration of the data
    """
    req_keys = ["mapper", "inner_reducer", "outer_reducer", "steps",
                "est_param"]
    _check_conf(cfg, req_keys, cat="method")
    return

def check_conf(cfg, cat="crossval", verbose=False):
    if cat == "crossval":
        _check_cv_conf(cfg)
    elif cat == "method":
        _check_method_conf(cfg)
    elif cat == "data":
        _check_data_conf(cfg)
    if verbose:
        print(cfg)

def JSONify_estimator(est, est_param, out=None,
            model_selection=True, path_to_mr="./"):
    """helper function to jsonify a sklearn.pipeline.Pipeline
    or an estimator.
    """
    import json
    import os.path as path
    from sklearn.pipeline import Pipeline

    steps = []
    if est.__class__ is Pipeline:
        for n, s in est.steps:
            t = [repr(s.__class__).split("\'")[1], s.__dict__]
            steps.append([n, t])
    else:
        cl = repr(est.__class__).split("\'")[1]
        n = cl.split(".")[-1]
        steps.append([n, [cl, est.__dict__]])
    conf = {}
    conf["steps" ] = steps
    conf["est_param"] = est_param
    conf["mapper"] = path.join(path_to_mr, "generic_mapper.py")
    conf["inner_reducer"] = path.join(path_to_mr, "generic_inner_reducer.py")
    conf["outer_reducer"] = path.join(path_to_mr, "generic_outer_reducer.py")
    if out is None:
        print(json.dumps(conf, indent=2))
    else:
        with open(out, 'w') as fd:
            json.dump(conf, fd, indent=2)
