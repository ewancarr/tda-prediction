# Re-run the pipeline manually to extract feature importances
# 2024-04-29

from pathlib import Path
from datetime import datetime
from joblib import load, dump
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer 
from sklearn.model_selection import GridSearchCV
from glmnet import LogitNet
from scipy.stats import ttest_ind
from functions import *
inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
replong, repwide = load(inp / 'repmea.joblib')
prs = load(inp / 'prs.joblib')
samp = load(inp / 'samp.joblib')

# Add prefix to repeated measures to make it easier to identify later
with_prefix = ['rep_' + i for i in repwide.columns]
repwide.columns = with_prefix

# Merge into single WIDE dataset
comb = baseline.merge(repwide, left_index=True, right_index=True, how='inner')
comb = comb.loc[set(outcomes.index).intersection(comb.index), :]
before = comb.copy()
n1 = np.shape(comb)[0]

# Remove people with less than 80% complete data among repeated measures
# at weeks 0, 1 and 2. (Week 0 = baseline).
r = comb.columns.str.contains('w[012]$')
incl = ((comb.loc[:, r].notna().sum(axis=1) / r.sum()) > 0.80)
pct = (comb.loc[:, r].notna().sum(axis=1) / r.sum())
incl = pct[pct > 0.8].index
comb = comb.loc[incl, :]
after = comb.copy()
n2 = np.shape(comb)[0]

# Select outcome
hdremit = outcomes.loc[incl, 'remit']

# Identify samples
opts = (('A', 'escitalopram', 'randomized'),
        ('B', 'nortriptyline', 'randomized'))
samp = {}
for o in opts:
    samp[o] = comb.loc[(comb['drug'] == o[1]) &
                       (comb['random'] == o[2]), ].index
samp[('C', 'both', 'anyrandom')] = comb.index
dump(samp, filename='data/samp.joblib')

for k, v in samp.items():
    print(k, 'n =', len(v))

# Recode 'drug'; remove 'random'
comb['escit'] = comb['drug'] == 'escitalopram'
comb.drop(labels=['drug', 'random'], axis=1, inplace=True)

# Prepare 'baseline' features
# Recode 'drug'; remove 'random'
baseline['escit'] = baseline['drug'] == 'escitalopram'
baseline.drop(labels=['drug', 'random'], axis=1, inplace=True)

config = {}
config['grid_search'] = 'saved/final/2021_08_08_153539_grid_search.joblib'
cv_inner = load(config['grid_search'])

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                 Refit 'best' parameters, with repeated CV                 ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

mrg = {'left_index': True, 'right_index': True, 'how': 'inner'}
use_last_week_only = True

def do_transform(X, f):
    return(pd.DataFrame(f.fit_transform(X),
                        index=X.index,
                        columns=X.columns))

def manual_pipe(X, y, w, growth_curves=False):
    # This is a rewrite of the Pipeline used elsewhere. It's needed so that we
    # can extract feature names (and therefore importances).
    names_input = list(X)
    # Step 1: Imputation
    knn = KNNImputer(n_neighbors=5)
    X = do_transform(X, knn)
    # Step 2: Growth curves (optionally)
    if growth_curves:
        X = fit_growth_curves(X, w)
    # Step 3: Zero variance
    names_zv = X.columns[np.var(X) > 0.0]
    X = X[names_zv]
    # Step 4: Scaler
    sc = StandardScaler()
    X = do_transform(X, sc)
    # Step 5: Estimator:
    crf = LogitNet(alpha=0.5,
                   cut_point=0,
                   n_splits=3,
                   random_state=42)
    crf.fit(X, y)
    prob = crf.predict_proba(X)
    features = pd.DataFrame(crf.coef_[0], index = X.columns, columns=['coef'])
    names_selected = features[abs(features.coef) > 0].index
    # Return
    return({'names_input': names_input,
           'names_zv': names_zv,
           'names_selected': names_selected,
           'n_selected': len(features[features.coef > 0]),
           'coef': features,
           'crf': crf,
            'y': y,
           'prob': prob})

apparent = {}
for k, v in samp.items():
    for max_week in [2, 4, 6]:
        for use_prs in [False, True]:
            print(k)
            # Prepare baseline features
            bl = baseline.loc[v].copy()
            if k[1] != 'both':
                bl.drop(labels=['escit'], axis=1, inplace=True)
            if use_prs:
                bl = bl.merge(prs,
                              how='left',
                              left_index=True,
                              right_index=True)
            rm = replong.loc[v].copy()
            rm = rm[rm['week'] <= max_week]
            rm['col'] = rm['variable'] + '__w' + rm['week'].astype('str')
            rm = rm[['col', 'value']].pivot(columns='col', values='value')
            rm = rm.add_prefix('rep__')
            params = cv_inner[(k, False, max_week)].best_params_['topo__kw_args']
            params['keep_rm'] = False
            params['max_week'] = max_week
            ls = compute_topological_variables(bl.merge(rm, **mrg).copy(), **params)
            ls = ls.loc[:, ls.columns.str.startswith('X')]
            ls.index.rename('subjectid', inplace=True)
            first_and_last = rm.columns.str.endswith('__w0') | rm.columns.str.endswith(f'__w{max_week}')
            rm_features = rm.loc[:, first_and_last]
            # Prepare outcome
            y = hdremit.loc[v].copy()
            # Option 1) Baseline only —————————————————————————————————————————
            i = ('1. Baseline only', k, max_week, use_prs)
            print(i)
            X = bl.copy()
            apparent[i] = manual_pipe(X, y, w=max_week)
            # Option 2) RM only ———————————————————————————————————————————————
            i = ('2. RM only', k, max_week, use_prs)
            print(i)
            X = bl.merge(rm_features, **mrg)
            apparent[i] = manual_pipe(X, y, w=max_week)
            # Option 3) RM + landscapes ———————————————————————————————————————
            i = ('3. RM + LS', k, max_week, use_prs)
            print(i)
            X = bl.merge(rm_features, **mrg).merge(ls, **mrg)
            apparent[i] = manual_pipe(X, y, w=max_week)
            # Option 3: GC only ———————————————————————————————————————————————
            i = ('4. GC only', k, max_week, use_prs)
            print(i)
            X = bl.merge(rm, **mrg)
            apparent[i] = manual_pipe(X, y, w=max_week, growth_curves=True)
            # Option 5: GC + landscapes ———————————————————————————————————————
            i = ('5. GC + LS', k, max_week, use_prs)
            print(i)
            X = bl.merge(rm, **mrg).merge(ls, **mrg)
            apparent[i] = manual_pipe(X, y, w=max_week, growth_curves=True)

dump(apparent, filename=tstamp('apparent'))
