# Title: Prediction models using repeated measures in GENDEP
# Author: Ewan Carr
# Started: 2021-06-25
# Updated: 2023-04-03

# import re
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

config = {}
# Re-run grid search for topological parameters? Almost never.
config['refit_grid_search'] = False
config['grid_search'] = 'saved/final/2021_08_08_153539_grid_search.joblib'
# Re-run internal validation? Almost always.
config['refit_iv'] = True
config['iv_path'] = 'saved/final/2023_05_22_150319_cv_results.joblib'
config['select_subsample'] = False
config['folds_inner'] = 10
config['folds_outer'] = 10
config['n_reps'] = 100
config['cores'] = 16

def tstamp(suffix):
    return(datetime.today().strftime('%Y_%m_%d_%H%M%S') +
             '_' + suffix + '.joblib')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                               Prepare data                                ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
replong, repwide = load(inp / 'repmea.joblib')
prs = load(inp / 'prs.joblib')

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

# Select subsample for testing purposes
if config['select_subsample']:
    for k, v in samp.items():
        samp[k] = pd.Series(v).sample(frac=0.3)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                       Compare included vs. excluded                       ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

print(f'Sample size before any exclusions: {n1}')
print(f'After excluding participants w/o repeated measures: {n2}')
print(f'Number excluded: {n1 - n2}')

before['incl'] = before.index.isin(after.index)

def tt(var):
    print(var, ttest_ind(before.loc[before['incl']][var],
                         before.loc[~before['incl']][var],
                         nan_policy='omit'),
          '\n',
          'incl:', np.mean(before.loc[before['incl']][var]),
          '\n',
          'excl:', np.mean(before.loc[~before['incl']][var]))

tt('age')
tt('educ')
tt('madrs')
tt('bdi')
tt('hdrs')
tt('bmi')

# Check prevalence by sample
for k, v in samp.items():
    print(k, hdremit.loc[v].mean())

# Export data as CSVs for debugging
baseline.to_csv('baseline.csv')
outcomes.to_csv('outcomes.csv')
prs.to_csv('prs.csv')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃          Define pipeline to tune persistence landscape variables          ┃
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
for land in [3, 5, 10, 12, 15]:
    for bins in [10, 100, 1000, 2000]:
        for mas in [2000, 1e5]:
            for dims in [[0], [0, 1], [0, 1, 2]]:
                param_grid.append({'topo__kw_args':
                                   [{'fun': 'landscape',
                                     'n_land': land,
                                     'bins': bins,
                                     'dims': dims,
                                     'mas': mas}
                                    ]
                                   })

#  ┌─────────────────────────────────────────────────────────┐ 
#  │                                                         │
#  │                Tune landscape parameters                │
#  │                                                         │
#  └─────────────────────────────────────────────────────────┘

# NOTE: using 10-fold CV without repetition.
# NOTE: takes a long time; typically we don't need to re-run this.

if config['refit_grid_search']:
    cv_inner = {}
    for k, v in samp.items():
        # For each sample [A, B, C]
        for max_week in [2, 4, 6]:
            # For increasing weeks of data
            for keep_rm in [True, False]:
                # With/without retaining the repeated measures in the model
                params = param_grid.copy()
                for p in params:
                    p['topo__kw_args'][0]['keep_rm'] = keep_rm
                    p['topo__kw_args'][0]['max_week'] = max_week
                X = comb.loc[v].copy()
                y = hdremit.loc[v].copy()
                if k[1] != 'both':
                    X.drop(labels=['escit'], axis=1, inplace=True)
                gs = GridSearchCV(pipe,
                                  param_grid,
                                  cv=10,
                                  scoring='roc_auc',
                                  n_jobs=-1,
                                  verbose=2)
                cv_inner[(k, keep_rm, max_week)] = gs.fit(X, y)
                del X
                del y
                del params
        dump(cv_inner, filename=tstamp('grid_search'))
else:
    cv_inner = load(config['grid_search'])

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                 Refit 'best' parameters, with repeated CV                 ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

'''
We're interested in the following options:

    1. Baseline only
    2. Repeated measures
    3. Repeated measures + landscapes
    4. Growth curves
    5. Growth curves + landscapes
'''

mrg = {'left_index': True, 'right_index': True, 'how': 'inner'}

# Option to decide whether to use the latest repeated measure only (i.e., week
# 4 only, for the '4 week' models), or all preceding measures.
use_last_week_only = True

# samp = {k: v for k, v in samp.items() if k[0] == 'B'}

cv_results = {}
# prs_results = {}
if config['refit_iv']:
    for k, v in samp.items():
        for max_week in [2, 4, 6]:
            for use_prs in [False, True]:
                # Prepare baseline features
                bl = baseline.loc[v].copy()
                if k[1] != 'both':
                    bl.drop(labels=['escit'], axis=1, inplace=True)
                if use_prs:
                    bl = bl.merge(prs,
                                  how='left',
                                  left_index=True,
                                  right_index=True)

                # Prepare repeated measures features
                rm = replong.loc[v].copy()
                rm = rm[rm['week'] <= max_week]
                rm['col'] = rm['variable'] + '__w' + rm['week'].astype('str')
                rm = rm[['col', 'value']].pivot(columns='col', values='value')
                rm = rm.add_prefix('rep__')

                # Prepare landscape features
                params = cv_inner[(k, False, max_week)].best_params_['topo__kw_args']
                params['keep_rm'] = False
                params['max_week'] = max_week
                ls = compute_topological_variables(bl.merge(rm, **mrg).copy(), **params)
                ls = ls.loc[:, ls.columns.str.startswith('X')]
                ls.index.rename('subjectid', inplace=True)

                # Decide: use all repeated measures or just latest assessment?
                if use_last_week_only:
                    first_and_last = rm.columns.str.endswith('__w0') | rm.columns.str.endswith(f'__w{max_week}')
                    rm_features = rm.loc[:, first_and_last]
                else:
                    rm_features = rm.copy()

                # Prepare outcome
                y = hdremit.loc[v].copy()

                # Set CV parameters
                cv_param = {'reps': config['n_reps'],
                            'cores': config['cores'],
                            'folds_inner': config['folds_inner'],
                            'folds_outer': config['folds_outer']
                            }

                # NOTE: We include baseline features in all models.

                # Option 1) Baseline only —————————————————————————————————————————
                i = ('1. Baseline only', k, max_week, use_prs)
                print(i)
                X = bl.copy()
                cv_results[i] = evaluate_model(X, y, **cv_param)

                # Option 2) RM only ———————————————————————————————————————————————
                i = ('2. RM only', k, max_week, use_prs)
                print(i)
                X = bl.merge(rm_features, **mrg)
                cv_results[i] = evaluate_model(X, y, **cv_param)

                # Option 3) RM + landscapes ———————————————————————————————————————
                i = ('3. RM + LS', k, max_week, use_prs)
                print(i)
                X = bl.merge(rm_features, **mrg).merge(ls, **mrg)
                cv_results[i] = evaluate_model(X, y, **cv_param)

                # Option 3: GC only ———————————————————————————————————————————————
                i = ('4. GC only', k, max_week, use_prs)
                print(i)
                X = bl.merge(rm, **mrg)
                # Run CV, incorporating growth curves
                cv_results[i] = evaluate_model(X, y,
                                               **cv_param,
                                               generate_curves=True)

                # Option 5: GC + landscapes ———————————————————————————————————————
                i = ('5. GC + LS', k, max_week, use_prs)
                print(i)
                X = bl.merge(rm, **mrg).merge(ls, **mrg)
                cv_results[i] = evaluate_model(X, y,
                                               **cv_param, 
                                               generate_curves=True)

if config['refit_iv']:
    dump(cv_results, filename=tstamp('cv_results'))


# Print the TDA parameters

tda_params = {}
for k, v in cv_inner.items():
    tda_params[k] = v.best_params_['topo__kw_args']

pd.DataFrame(tda_params).to_excel('tables/tda_params.xlsx')

#  END
