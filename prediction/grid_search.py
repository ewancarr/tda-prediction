# Title: Tune landscape parameters using scikit-learn
# Author: Ewan Carr
# Started: 2021-06-25

from pathlib import Path
from joblib import load
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
import gudhi as gd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import (RepeatedKFold, GridSearchCV,
                                     cross_val_score, cross_validate, KFold)
from gudhi.representations import (DiagramSelector, Clamping, Landscape,
                                   Silhouette, BettiCurve)
from glmnet import LogitNet

from functions import (knn,
                       compute_topological_variables,
                       reshape)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                               Prepare data                                ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
_, repwide = load(inp / 'repmea.joblib')

# Add prefix to repeated measures to make it easier to identify later
with_prefix = ['rep_' + i for i in repwide.columns]
repwide.columns = with_prefix

# Merge into single WIDE dataset
comb = baseline.merge(repwide, left_index=True, right_index=True, how='inner')
comb = comb.loc[set(outcomes.index).intersection(comb.index), :]

# Remove people with at least 80% complete data among repeated measures
# at weeks 0, 1 and 2. (Week 0 = baseline).
r = comb.columns.str.contains('w[012]$')
incl = ((comb.loc[:, r].notna().sum(axis=1) / r.sum()) > 0.80)
pct = (comb.loc[:, r].notna().sum(axis=1) / r.sum())
incl = pct[pct > 0.8].index

incl = np.random.choice(incl, size=100, replace=False)

comb = comb.loc[incl, :]

# Select outcome
y = outcomes.loc[incl, 'remit']

# Identify samples
opts = (('A', 'escitalopram', 'randomized'),
        ('B', 'nortriptyline', 'randomized'))
samp = {}
for o in opts:
    samp[o] = comb.loc[(comb['drug'] == o[1]) &
                       (comb['random'] == o[2]), ].index
samp[('C', 'both', 'anyrandom')] = comb.index

for k, v in samp.items():
    print(k, 'n =', len(v))

# Recode 'drug'; remove 'random'
comb['talopram'] = comb['drug'] == 'escitalopram'
comb.drop(labels=['drug', 'random'], axis=1, inplace=True)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                      Define pipeline and parameters                       ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

pipe = Pipeline(steps=[
    ('impute', FunctionTransformer(knn)),
    ('topo', FunctionTransformer(compute_topological_variables)),
    ('zerovar', VarianceThreshold()),
    ('estimator', LogitNet(alpha=0.5,
                           cut_point=0,
                           n_splits=10,
                           random_state=42))])

param_grid = []
for land in [1, 3, 5, 10, 15]:
    for bins in [10, 100, 1000, 2000]:
        for mas in [10, 100, 1000, 1e5]:
            for dims in [[0], [0, 1], [0, 1, 2]]:
                param_grid.append({'topo__kw_args': [{'fun': 'landscape',
                                                      'n_land': land,
                                                      'bins': bins,
                                                      'dims': dims,
                                                      'mas': mas}]})


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                            Run inner/outer CV                             ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

cv_inner = KFold(n_splits=2, shuffle=True, random_state=42)
cv_outer = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)

param_grid = param_grid[:5]

gs = GridSearchCV(pipe,
                  param_grid,
                  cv=cv_inner,
                  scoring='roc_auc',
                  n_jobs=-1)


fit = cross_validate(gs,
                     comb,
                     y,
                     cv=cv_outer,
                     n_jobs=-1,
                     scoring='roc_auc',
                     verbose=2)
