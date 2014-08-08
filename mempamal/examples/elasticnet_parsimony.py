from parsimony.estimators import ElasticNet
from sklearn.base import BaseEstimator


class EnetWrap(BaseEstimator, ElasticNet):

    def __init__(self, l=0., alpha=1., algorithm_params={},
                 penalty_start=0, mean=True):
        super(EnetWrap, self).__init__(l=l, alpha=alpha,
                                       algorithm_params=algorithm_params,
                                       penalty_start=penalty_start, mean=mean)
