# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause

from sklearn.datasets import load_iris

def get_data(**kwargs):
    data = load_iris()
    x = data['data']
    y = data['target']
    return x, y
