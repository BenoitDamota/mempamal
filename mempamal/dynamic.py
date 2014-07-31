# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause

def dynamic_import(str_import):
    mod, cla = str_import.rsplit('.', 1)
    dyn_import = getattr(__import__(mod, fromlist=[str(cla)]), cla)
    return dyn_import
    
def get_step(step):
    str_imp = step[1][0]
    kwargs = step[1][1]
    name = step[0]
    return (name, (dynamic_import(str_imp), kwargs))

def construct_pipeline(cfg):
    steps = cfg["steps"]
    est_param = cfg["est_param"]
    pipe = []
    for step in steps:
        pipe.append(get_step(step))
    return {'steps': pipe}, est_param

def get_score_func(cfg, cv="crossval_score"):
    """Load the score function from the CV configuration.

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
    func = cfg['func']
    kwargs = cfg['kwargs']
    mod, cla = func.rsplit('.', 1)
    dyn_func = getattr(__import__(mod, fromlist=[str(cla)]), cla)
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
    y : array, shape (n_samples, n_features)
    """
    func = cfg["gridSearch"]['parametersGrid'][0]
    kwargs = cfg["gridSearch"]['parametersGrid'][1]
    mod, cla = func.rsplit('.', 1)
    dyn_func = getattr(__import__(mod, fromlist=[str(cla)]), cla)
    grid = dyn_func(x, y, **kwargs)
    return grid

