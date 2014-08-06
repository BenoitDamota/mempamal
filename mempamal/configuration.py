# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Functions relative to configurations.
"""
import json
import os.path as path
import warnings

import sklearn.externals.joblib as joblib
from sklearn.pipeline import Pipeline

from mempamal.dynamic import get_grid
from mempamal.crossval import make_folds


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
    req_keys = ["mapper", "inner_reducer", "outer_reducer", "steps"]
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


def JSONify_estimator(est,
                      model_selection=False,
                      param_val=None,
                      out=None,
                      path_to_mr=None,
                      mapper="mapper.py",
                      i_red="inner_reducer.py",
                      o_red="outer_reducer.py"):
    """Helper function to jsonify a sklearn.pipeline.Pipeline or an estimator.

    Parameters
    ----------
    est : estimator,
        Estimator to jsonify.
    model_selection : boolean, optional (default=True)
        Do you plan to perform a model selection (optimization of
        hyper-parameters)?
    param_val : val, optional (default=None)
        If model_selection is False, you should provide a value for
        the est_param.
    out : str, optional (default=None)
        Filename to output the json.
    path_to_mr : str, optional (default=None, i.e. local mempamal directory)
        Where to find the mapper and reducers scripts.
    mapper : str, optional (default="mapper.py")
        script for the mapper
    i_red : str, optional (default="inner_reducer.py")
        script for the inner reducer (for model selection)
    o_red : str, optional (default="outer_reducer.py")
        script for the outer reducer

    Examples:
    ---------
    >>> from sklearn.linear_model.logistic import LogisticRegression
    >>> from sklearn.preprocessing.data import StandardScaler
    >>> from sklearn.pipeline import Pipeline
    >>> from mempamal.configuration import JSONify_estimator
    >>> from json import dumps
    >>> s1 = StandardScaler(with_mean=True, with_std=False)
    >>> s2 = LogisticRegression()
    >>> print(dumps(JSONify_estimator(Pipeline([("scaler", s1),
            ("logit", s2)]), "logit__C", path_to_mr="."), indent=2))
{
  "inner_reducer": "./inner_reducer.py",
  "mapper": "./mapper.py",
  "steps": [
    [
      "scaler",
      [
        "sklearn.preprocessing.data.StandardScaler",
        {
          "copy": true,
          "with_mean": true,
          "with_std": false
        }
      ]
    ],
    [
      "logit",
      [
        "sklearn.linear_model.logistic.LogisticRegression",
        {
          "loss": "lr",
          "C": 1.0,
          "verbose": 0,
          "dual": false,
          "fit_intercept": true,
          "penalty": "l2",
          "multi_class": "ovr",
          "random_state": null,
          "tol": 0.0001,
          "class_weight": null,
          "intercept_scaling": 1
        }
      ]
    ]
  ],
  "outer_reducer": "./outer_reducer.py",
  "est_param": "logit__C"
}
    """
    if path_to_mr is None:
        import mempamal
        path_to_mr = path.join(path.dirname(mempamal.__file__), "scripts")
    # introspection of the estimator
    steps = []
    if est.__class__ is Pipeline:
        # if the estimator is a Pipeline, retrieve all steps names,
        # classes and dict
        for n, s in est.steps:
            t = [repr(s.__class__).split("\'")[1], s.get_params()]
            steps.append([n, t])
    else:
        # else the name of the unique step is the name of the
        # estimator class
        cl = repr(est.__class__).split("\'")[1]
        n = cl.split(".")[-1]
        steps.append([n, [cl, est.get_params()]])

    # produce a method configuration
    conf = {}
    conf["steps"] = steps
    conf["mapper"] = path.join(path_to_mr, mapper)
    conf["inner_reducer"] = path.join(path_to_mr, i_red)
    conf["outer_reducer"] = path.join(path_to_mr, o_red)

    check_conf(conf, cat="method")
    # output
    if out is not None:
        conf["src"] = path.basename(out)
        with open(out, 'w') as fd:
            json.dump(conf, fd, indent=2)
    return conf


def JSONify_cv(cv, score_func,
               cv_kwargs=None,
               score_func_kwargs=None,
               inner_cv=None,
               inner_cv_kwargs=None,
               inner_score_func=None,
               inner_score_func_kwargs=None,
               stratified=False,
               grid_func=None,
               grid_func_kwargs=None,
               out=None):
    """Helper function to create a cross-validation configuration

    parameters:
    -----------
    cv : class,
        foldsIterator for the crossval_score (outer CV).
    score_func : func,
        Scoring function for the crossval_score.
    cv_kwargs : dict, optional (default=None),
        Keywords argument for cv.
    score_func_kwargs : dict, optional (default=None),
        Keywords argument for score_func.
    inner_cv : class, optional (default=None),
        foldsIterator for the gridSearch (inner CV).
    inner_cv_kwargs : dict, optional (default=None),
        Keywords argument for inner_cv.
    inner_score_func : function, optional (default=None),
        Scoring function for the gridSearch
    inner_score_func_kwargs : dict, optional (default=None),
        Keywords argument for inner_score_func.
    stratified : boolean, optional (default=False),
        Are the foldsIterators stratified.
    grid_func : function, optional (default=None),
        Function to compute a grid of parameters.
    grid_func_kwargs : dict, optional (default=None),
        Keywords argument for grid_func.
    out : str, optional (default=None)
        Filename to output the json.

    Examples:
    ---------
    >>> from sklearn.cross_validation import StratifiedShuffleSplit
    >>> from sklearn.cross_validation import StratifiedKFold
    >>> from sklearn.metrics import f1_score
    >>> from mempamal.configuration import JSONify_cv
    >>> from mempamal.examples.parameters_grid import make_log_grid
    >>> from json import dumps
    >>> print(dumps(JSONify_cv(StratifiedShuffleSplit,
                    cv_kwargs={"test_size": 0.2, "random_state": 42},
                    score_func=f1_score,
                    score_func_kwargs={"average": "weighted"},
                    inner_cv=StratifiedKFold,
                    inner_cv_kwargs={"n_folds": 5},
                    inner_score_func=f1_score,
                    inner_score_func_kwargs={"average": "weighted"},
                    stratified=True,
                    grid_func=make_log_grid), indent=2))
{
  "crossval_score": {
    "funcMetric": [
      "sklearn.metrics.metrics.f1_score",
      {
        "average": "weighted"
      }
    ],
    "foldsIterator": [
      "sklearn.cross_validation.StratifiedShuffleSplit",
      {
        "test_size": 0.2,
        "random_state": 42
      }
    ]
  },
  "modelSelection": true,
  "stratified": true,
  "gridSearch": {
    "parametersGrid": [
      "mempamal.examples.parameters_grid.make_log_grid",
      {}
    ],
    "funcMetric": [
      "sklearn.metrics.metrics.f1_score",
      {
        "average": "weighted"
      }
    ],
    "foldsIterator": [
      "sklearn.cross_validation.StratifiedKFold",
      {
        "n_folds": 5
      }
    ]
  }
}
    """
    conf = {}
    cv_kwargs = ({} if cv_kwargs is None else cv_kwargs)
    if (inner_cv is not None) and (inner_score_func is None):
        raise TypeError("inner_score_func is None where"
                        " a function is required.")
    if (inner_cv is not None) and (grid_func is None):
        raise TypeError("grid_func is None where"
                        " a function is required.")
    modelSelection = False if inner_cv is None else True
    # retrieve crossval_score object
    cv_cl = ".".join([cv.__module__, cv.__name__])
    sf = ".".join([score_func.__module__, score_func.__name__])
    sf_kwargs = ({} if score_func_kwargs is None
                 else score_func_kwargs)
    if modelSelection:
        icv_cl = ".".join([inner_cv.__module__, inner_cv.__name__])
        icv_kwargs = ({} if inner_cv_kwargs is None else inner_cv_kwargs)
        isf = ".".join([inner_score_func.__module__,
                        inner_score_func.__name__])
        isf_kwargs = ({} if inner_score_func_kwargs is None
                      else inner_score_func_kwargs)
        gf = ".".join([grid_func.__module__,
                       grid_func.__name__])
        gf_kwargs = ({} if grid_func_kwargs is None
                     else grid_func_kwargs)
    # produce a cv configuration
    conf["modelSelection"] = modelSelection
    conf["stratified"] = stratified
    conf["crossval_score"] = {}
    conf["crossval_score"]["foldsIterator"] = [cv_cl, cv_kwargs]
    conf["crossval_score"]["funcMetric"] = [sf, sf_kwargs]
    if modelSelection:
        conf["gridSearch"] = {}
        conf["gridSearch"]["foldsIterator"] = [icv_cl, icv_kwargs]
        conf["gridSearch"]["funcMetric"] = [isf, isf_kwargs]
        conf["gridSearch"]["parametersGrid"] = [gf, gf_kwargs]

    check_conf(conf, cat="crossval")
    # output
    if out is not None:
        conf["src"] = path.basename(out)
        with open(out, 'w') as fd:
            json.dump(conf, fd, indent=2)
    return conf


def build_dataset(X, y, method_conf, cv_conf,
                  outputdir=".",
                  verbose=False,
                  compress=0):
    """Write the dataset file.
    """
    n_samples = X.shape[0]
    n_targets = 1 if (y.ndim == 1) else y.shape[1]
    if n_targets > 1:
        warnings.warn("More than one target. Unexpected results or crashes "
                      "may occur if your methods and/or metrics "
                      "cannot handle multiple targets.", RuntimeWarning)
    output_file = path.join(outputdir, "dataset.joblib")
    folds = dict(make_folds(y, cv_conf, verbose=verbose),
                 src=path.basename(output_file))

    grid = (get_grid(cv_conf, X, y) if cv_conf["modelSelection"]
            else None)
    if verbose:
        print("Input dataset destination: {}".format(output_file))
    dataset = {"X": X, "Y": y,
               "n_samples": n_samples, "n_targets": n_targets,
               "folds": folds, "grid": grid}
    joblib.dump(dataset, output_file, compress=compress)
    return dataset
