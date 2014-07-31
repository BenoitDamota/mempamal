# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Functions relative to configurations.
"""
import json
import os.path as path

from sklearn.pipeline import Pipeline

def _check_conf(cfg, req_keys, cat=""):
    """Generic checking for required keys.

    """
    for k in req_keys:
        try:
            cfg[k]
        except KeyError:
            raise KeyError(("Required key is missing in the "
                            "{} configuration file: {}").format(cat, k))

def _check_cv_conf(cfg):
    """Check the configuration of the cross-validation.

    """
    req_keys = ["stratified", "modelSelection", "crossval_score"]
    _check_conf(cfg, req_keys, cat="crossval")
    req_keys = ["foldsIterator", "funcMetric"]
    _check_conf(cfg["crossval_score"], req_keys, cat="crossval")
    if cfg["modelSelection"]:
        req_keys = ["foldsIterator", "funcMetric", "parametersGrid"]
        _check_conf(cfg, ["gridSearch"], cat="crossval")
        _check_conf(cfg["gridSearch"], req_keys, cat="crossval")

def _check_data_conf(cfg):
    """Check the configuration of the data.

    """
    req_keys = ["func", "kwargs", "in_out_dir"]
    _check_conf(cfg, req_keys, cat="data")

def _check_method_conf(cfg):
    """Check the configuration of the data.

    """
    req_keys = ["mapper", "inner_reducer", "outer_reducer", "steps",
                "est_param"]
    _check_conf(cfg, req_keys, cat="method")

def check_conf(cfg, cat="crossval", verbose=False):
    """Entry point to check a configuration by category.

    Parameters
    ----------
    cfg : dict,
        the configuration to check.
    cat : str in [crossval, method, data], optional (default="crossval")
        category of the configuration to call the good checking method.
    verbose : boolean, optional (default=False)
        verbose mode.
    """
    if cat == "crossval":
        _check_cv_conf(cfg)
    elif cat == "method":
        _check_method_conf(cfg)
    elif cat == "data":
        _check_data_conf(cfg)
    if verbose:
        print(cfg)

def JSONify_estimator(est, est_param, 
                      model_selection=True, 
                      param_val=None,
                      out=None, 
                      path_to_mr=".",
                      mapper="mapper.py",
                      i_red="inner_reducer.py",
                      o_red="outer_reducer.py"):
    """Helper function to jsonify a sklearn.pipeline.Pipeline or an estimator.

    Parameters
    ----------
    est : estimator,
        Estimator to jsonify.
    est_param : str,
        Parameter to optimize.
    model_selection : boolean, optional (default=True)
        Do you plan to perform a model selection (optimization of 
        hyper-parameters)?
    param_val : val, optional (default=None)
        If model_selection is False, you should provide a value for
        the est_param.
    out : str, optional (default=None)
        Filename to output the json, if None the json is printed on stdout.
    path_to_mr : str, optional (default=".", i.e. working directory)
        Where to find the mapper and reducers scripts.
    mapper : str, optional (default="mapper.py")
        script for the mapper
    i_red : str, optional (default="inner_reducer.py")
        script for the inner reducer (for model selection)
    o_red : str, optional (default="outer_reducer.py")
        script for the outer reducer
    """
    # introspection of the estimator
    steps = []
    if est.__class__ is Pipeline:
        # if the estimator is a Pipeline, retrieve all steps names,
        # classes and dict
        for n, s in est.steps:
            t = [repr(s.__class__).split("\'")[1], s.__dict__]
            steps.append([n, t])
    else:
        # else the name of the unique step is the name of the
        # estimator class
        cl = repr(est.__class__).split("\'")[1]
        n = cl.split(".")[-1]
        steps.append([n, [cl, est.__dict__]])

    # produce a  method configuration
    conf = {}
    conf["steps" ] = steps
    conf["est_param"] = est_param
    if model_selection is False:
        conf[est_param] = param_val
    conf["mapper"] = path.join(path_to_mr, mapper)
    conf["inner_reducer"] = path.join(path_to_mr, i_red)
    conf["outer_reducer"] = path.join(path_to_mr, o_red)

    # output
    if out is None:
        print(json.dumps(conf, indent=2))
    else:
        with open(out, 'w') as fd:
            json.dump(conf, fd, indent=2)
