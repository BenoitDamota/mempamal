# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Functions relative to grid search.
"""
import numpy as np


def make_log_grid(x, y, n_param=10, pmin=-4, pmax=5):
    params = np.logspace(pmin, pmax, n_param)[::-1]
    grid = []
    for p in params:
        grid.append({"logit__C": p})
    return grid
