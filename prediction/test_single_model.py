import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from joblib import load, dump
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import recall_score
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import (make_scorer, roc_auc_score, confusion_matrix,
                             recall_score, brier_score_loss)
from sklearn import ensemble
from sklearn.preprocessing import scale, StandardScaler
from sklearn.feature_selection import VarianceThreshold

def tn(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[0, 0])


def fp(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[0, 1])


def fn(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[1, 0])


def tp(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[1, 1])


def calc_npv(y_true, y_prob, threshold=0.5):
    y_pred = y_prob > threshold
    n_tn = int(tn(y_true, y_pred))
    n_fn = int(fn(y_true, y_pred))
    if n_tn == 0:
        npv = np.nan
    else:
        with np.errstate(invalid='ignore'):
            npv = np.mean(n_tn / (n_tn + n_fn))
    return(npv)


def calc_ppv(y_true, y_prob, threshold=0.5):
    y_pred = y_prob > threshold
    n_tp = int(tp(y_true, y_pred))
    n_fp = int(fp(y_true, y_pred))
    if n_tp == 0:
        ppv = np.nan
    else:
        with np.errstate(invalid='ignore'):
            ppv = np.mean(n_tp / (n_tp + n_fp))
    return(ppv)


def sens(y_true, y_prob, threshold):
    y_pred = y_prob > threshold
    return(recall_score(y_true, y_pred))


def spec(y_true, y_prob, threshold):
    y_pred = y_prob > threshold
    return(recall_score(y_true, y_pred, pos_label=0))

scorers = {'auc': 'roc_auc',
           'sens': make_scorer(recall_score),
           'spec': make_scorer(recall_score, pos_label=0),
           'ppv': make_scorer(calc_ppv, needs_proba=True),
           'npv': make_scorer(calc_npv, needs_proba=True),
           'tp': make_scorer(tp),
           'tn': make_scorer(tn),
           'fp': make_scorer(fp),
           'fn': make_scorer(fn),
           'brier': make_scorer(brier_score_loss,
                                greater_is_better=False,
                                needs_proba=True)}

i = sys.argv[1]
n_rep = int(sys.argv[2])

X = load('prediction/sets/' + str(i))['data']
y = load('data/outcomes.joblib')[['remit']]

Xy = X.merge(y,
             left_index=True,
             right_index=True,
             how='left')         # 'left' is important here. It ensures we only
                                 # include outcome information for the sample
                                 # defined in X.

X = Xy.drop(labels='remit', axis=1)
y = Xy['remit'].values

# Fit in Python/Scikit

clf_rf = make_pipeline(VarianceThreshold(),
                       StandardScaler(),
                       KNNImputer(n_neighbors=5),
                       ensemble.RandomForestClassifier())

clf_lr = make_pipeline(VarianceThreshold(),
                       StandardScaler(),
                       KNNImputer(n_neighbors=5),
                       LogitNet())

rkf = RepeatedKFold(n_splits=10, 
                    n_repeats=n_rep, 
                    random_state=42)

fit = {}
for f, label in zip([clf_rf, clf_lr],
        ['random_forest', 'logit_net']):
    fit[label] = cross_validate(f,
                                X=X,
                                y=y,
                                scoring=scorers,
                                cv=rkf,
                                n_jobs=-1)

dump(fit, filename = 'prediction/fits/' + str(i))

# Fit in R/glmnet

CONTINUE HERE

X.to_csv('prediction/scratch/' + str(i) + '_X.csv')
pd.DataFrame({'y': y}).to_csv('prediction/scratch/' + str(i) + '_y.csv')
