# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Example of dataset and how to load its data.
"""
from sklearn.datasets import load_iris

def get_data(**kwargs):
    """Load data and return a x, y.

    Note: it ignores all arguments.
    """
    data = load_iris()
    x = data['data']
    y = data['target']
    return x, y
