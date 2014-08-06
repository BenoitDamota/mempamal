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

# create a simple pipeline with a StandardScaler and a LogisticRegression
s1 = StandardScaler(with_mean=True, with_std=False)
s2 = LogisticRegression()
p = [("scaler", s1), ("logit", s2)]
est = Pipeline(p)

# get the iris dataset
X, y = iris.get_data()

# jsonify the method and a cross-validation scheme
method_conf = JSONify_estimator(est, out="./est.json")
cv_conf = JSONify_cv(StratifiedKFold, cv_kwargs={"n_folds": 5},
                     score_func=f1_score, stratified=True,
                     out="./cv.json")
# build the dataset file
dataset = build_dataset(X, y, method_conf, cv_conf, ".", compress=1)

# create the workflow in the internal representation
wfi = create_wf(dataset['folds'], cv_conf, method_conf, ".",
               verbose=True)
# save to soma-workflow format
wf = save_wf(wfi, "./workflow.json", mode="soma-workflow")

# create a controler and submit
controler = WorkflowController()
wf_id = controler.submit_workflow(workflow=wf, name="first example")

# wait for completion
while controler.workflow_status(wf_id) != 'workflow_done':
    time.sleep(2)
# read final result file
print(joblib.load('./final_res.pkl'))
