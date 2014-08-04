import time

import sklearn.externals.joblib as joblib

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing.data import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from soma_workflow.client import WorkflowController

from mempamal.configuration import JSONify_estimator, JSONify_cv, build_dataset
from mempamal.workflow import create_wf, save_wf
from mempamal.datasets import iris

s1 = StandardScaler(with_mean=True, with_std=False)
s2 = LogisticRegression()
p = [("scaler", s1), ("logit", s2)]
est = Pipeline(p)

X, y = iris.get_data()

method_conf = JSONify_estimator(est, "logit__C", out="./est.json")
cv_conf = JSONify_cv(StratifiedKFold, cv_kwargs={"n_folds": 5},
                     score_func=f1_score, stratified=True,
                     out="./cv.json")

dataset = build_dataset(X, y, method_conf, cv_conf, ".")

wf = create_wf(dataset['folds'], cv_conf, method_conf, ".",
               verbose=True)
wf = save_wf(wf, "./workflow.json", mode="soma-workflow")

controler = WorkflowController()
wf_id = controler.submit_workflow(workflow=wf, name="first example")

while controler.workflow_status(wf_id) != 'workflow_done':
    time.sleep(2)
print(joblib.load('./final_res.pkl'))
