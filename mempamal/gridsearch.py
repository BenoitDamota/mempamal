# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
"""
Simple GridSearch for a pipelined estimator (without warm restart).
"""
import numpy as np


class GenericGridSearch(object):
    """Simple GridSearch for a pipelined estimator.

    Note: see sklearn.pipeline.Pipeline
    """

    def __init__(self, est, est_param, params, score_func,
                 est_kwargs=None,
                 score_kwargs=None):
        """

        Parameters
        ----------
        est : estimator,
            Estimator to jsonify.
        est_param : str,
            Parameter to optimize.
        params : array, shape(n_parameters)
            Grid of parameters
        score_func : func,
            scoreng function (e.g. sklearn.metrics.f1_score)
        est_kwargs : dict, optional (default=None)
            keywords arguments for the estimator
        score_kwargs : dict, optional (default=None)
            keywords arguments for the scoring function
        """
        self.params = params
        self.est_param = est_param
        self.est = est
        self.res = {}
        self.score_func = score_func
        if score_kwargs is not None:
            self.score_kwargs = score_kwargs
        else:
            self.score_kwargs = {}
        if est_kwargs is not None:
            self.est_kwargs = est_kwargs
        else:
            self.est_kwargs = {}

    def fit(self, X, y):
        """Fit the estimator on each parameter of the grid.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            features array
        y : array, shape (n_samples, n_targets)
            targets array
        """
        for p in self.params:
            param_kwargs = {self.est_param: p}
            steps = self.est_kwargs['steps']
            pipe_steps = []
            for s in steps:
                s_i = (s[1][0])()
                for k, v in (s[1][1]).iteritems():
                    s_i.__setattr__(k, v)
                pipe_steps.append((s[0], s_i))

            self.res[p] = self.est(pipe_steps)
            self.res[p].set_params(**param_kwargs)
            self.res[p].fit(X, y)

    def predict(self, X):
        """Predict the targets from X for each parameter of the grid

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            features array
        """
        y_pred = []
        for p in self.params:
            y_pred.append(self.res[p].predict(X))
        return np.asarray(y_pred)

    def score(self, y_test, y_pred):
        """Apply the scoring function for each prediction

        Parameters
        ----------
        y_test : array, shape (n_samples, n_targets)
            Real targets values
        y_pred : array, shape (n_parameters, n_samples, n_targets)
            Targets prediction to score.
        """
        scores = []
        for yp in y_pred:
            scores.append(self.score_func(y_test, yp, **self.score_kwargs))
        return np.asarray(scores).T
