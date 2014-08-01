# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause


def dynamic_import(str_import):
    """Take a string representing a python function or class and import it.

    Parameters:
    -----------
    str_import : str
        the string representing the import (e.g. "sklearn.metrics.f1_score")
    """
    mod, cla = str_import.rsplit('.', 1)
    dyn_import = getattr(__import__(mod, fromlist=[str(cla)]), cla)
    return dyn_import


def _get_step(step):
    """Import class of a given step and return a sklearn.pipeline.Pipeline step
    """
    str_imp = step[1][0]
    kwargs = step[1][1]
    name = step[0]
    return (name, (dynamic_import(str_imp), kwargs))


def construct_pipeline(cfg):
    """Construct the pipeline steps for sklearn.pipeline.Pipeline.

    Construct the pipeline steps for sklearn.pipeline.Pipeline and
    return the steps and the parameter to optimize (e.g. "logit__C" for
    the parameter "C" of the step named "logit").

    Parameters:
    -----------
    cfg : dict,
        method configuration describing the steps of an pipelined estimator
    """
    steps = cfg["steps"]
    est_param = cfg["est_param"]
    pipe = []
    for step in steps:
        pipe.append(_get_step(step))
    return {'steps': pipe}, est_param


def get_score_func(cfg, cv="crossval_score"):
    """Import score function and kwargs from the CV configuration.

    Parameters
    ----------
    cfg : dict,
        configuration dict for cross-validation.

    cv : str, optional(default="crossval_score")
        section of the configuration dict to search for a score function.
    """
    func = dynamic_import(cfg[cv]["funcMetric"][0])
    kwargs = cfg[cv]["funcMetric"][1]
    return func, kwargs


def load_data(cfg):
    """Load data from the data configuration.

    Parameters
    ----------
    cfg : dict,
        configuration dict for data and I/O.
    """
    kwargs = cfg['kwargs']
    dyn_func = dynamic_import(cfg['func'])
    x, y = dyn_func(**kwargs)
    return x, y


def get_grid(cfg, x, y):
    """Load the parameters grid from the CV configuration.

    Parameters
    ----------
    cfg : dict,
        configuration dict for cross-validation.
    x : array, shape (n_samples, n_features)
        features array
    y : array, shape (n_samples, n_targets)
        targets array
    """
    kwargs = cfg["gridSearch"]['parametersGrid'][1]
    dyn_func = dynamic_import(cfg["gridSearch"]['parametersGrid'][0])
    grid = dyn_func(x, y, **kwargs)
    return grid
