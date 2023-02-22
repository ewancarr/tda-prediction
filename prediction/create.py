# Title: Tune landscape parameters using scikit-learn
# Author: Ewan Carr
# Started: 2021-06-25

# from mpi4py import MPI
import re
from pathlib import Path
from datetime import datetime
from joblib import load, dump
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.manifold import MDS
from sklearn.model_selection import (GridSearchCV,
        KFold, RepeatedKFold, cross_validate)
from glmnet import LogitNet
import gudhi as gd
from gudhi.representations import DiagramSelector, Landscape
from scipy.stats import ttest_ind
from sklearn.metrics import (make_scorer, confusion_matrix,
                             recall_score, brier_score_loss,
                             accuracy_score,
                             balanced_accuracy_score)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from functions import *

# Re-run grid search? Almost never.
refit_grid_search = False
# Re-run internal validation? Almost always.
refit_iv_landscapes = True
select_subsample = False
folds_inner = 5
folds_outer = 5
n_reps = 50
cores = 16
grid_search = 'saved/2021_08_09/2021_08_08_153539_grid_search.joblib'

def tstamp(suffix):
    return(datetime.today().strftime('%Y_%m_%d_%H%M%S') + '_' + suffix + '.joblib')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                               Prepare data                                ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
replong, repwide = load(inp / 'repmea.joblib')

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
dump(samp, filename='samp.joblib')

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
if select_subsample:
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

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                     Load growth curve / RM parameters                     ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# alts = {}
# for mw in [2, 4, 6]:
#     g = load('prediction/re_params/re_' + str(mw))
#     g.columns = [i[0] + '_' + i[1].split('_')[1] for i in g.columns]
#     g = baseline.merge(g, left_index=True, right_index=True, how='inner')
#     alts['gc_' + str(mw)] = g

# def reshape_rm(d, mw):
#     d = d[d['week'] <= mw].copy()
#     d['col'] = d['variable'] + '_w' + d['week'].astype('str')
#     return(d[['col', 'value']].pivot(columns='col', values='value'))

# for mw in [2, 4, 6]:
#     r = reshape_rm(replong, mw)
#     alts['rm_' + str(mw)] = baseline.merge(r,
#             left_index=True,
#             right_index=True,
#             how='inner')

# print(list(alts))


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

# NOTE: This is using 10-fold CV without repetition.

if refit_grid_search:
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
    cv_inner = load(grid_search)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                 Refit 'best' parameters, with repeated CV                 ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

'''
We're interested in the following options:

    1. Repeated measures
    2. Repeated measures + landscapes
    3. Growth curves
    4. Growth curves + landscapes
'''

def fit_growth_curves(X):
    import re
    import numpy as np
    # Identify repeated measures
    pattern = re.compile("^rep__")
    repeated_measures = [bool(pattern.match(i)) for i in list(X)]

    # Separate baseline vs. repeated measures columns
    other = X.loc[:, [not i for i in repeated_measures]]
    X = X.loc[:, repeated_measures]

    # Select repeated measures, reshape to LONG format
    X = X.melt(ignore_index=False)
    X['week'] = X['variable'].str[-1:].astype('int')
    X.reset_index(inplace=True)

    # For each outcome, fit a growth curve
    outcomes = set([re.search(r"^rep__(.*)__w.*", f).group(1) 
                for f in X['variable'].unique()])

    random_effects = []
    for o in outcomes:
        df = X[X.variable.str.contains(o)].dropna()
        mdf = smf.mixedlm('value ~ week + np.power(week, 2)',
                          df,
                          groups=df['subjectid'], 
                          re_formula='~ week + np.power(week, 2)')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            mdf = mdf.fit()
        re = pd.DataFrame(mdf.random_effects).T
        re.columns = [o + i for i in ['_int', '_t', '_t2']]
        random_effects.append(re)
    # Append all random effects; merge with 'baseline' variables
    return(other.merge(pd.concat(random_effects, axis=1),
                       left_index=True,
                       right_index=True))


def evaluate_model(X, y,
                   generate_curves=False,
                   reps=50,
                   impute=False, # Don't think this is having any effect
                   return_estimator=False):
    features = X.columns
    # TODO: find cleaner way of having an optional step in the pipeline
    if generate_curves:
        pipe = Pipeline(steps=[
            ('impute', FunctionTransformer(knn)),
            ('growthcurves', GenerateGrowthCurves()),
            ('zerovar', VarianceThreshold()),
            ('estimator', LogitNet(alpha=0.5,
                                   cut_point=0,
                                   n_splits=folds_inner,
                                   random_state=42))])
    else:
        pipe = Pipeline(steps=[
            ('impute', FunctionTransformer(knn)),
            ('zerovar', VarianceThreshold()),
            ('estimator', LogitNet(alpha=0.5,
                                   cut_point=0,
                                   n_splits=folds_inner,
                                   random_state=42))])
    rkf = RepeatedKFold(n_splits=folds_outer, n_repeats=reps, random_state=42)
    # rkf = KFold(n_splits=3) # 
    fit = cross_validate(pipe,
                         X=X,
                         y=y,
                         cv=rkf,
                         scoring=scorers,
                         n_jobs=cores,
                         return_estimator=return_estimator)
    # Refit, once, to get feature importance
    single = pipe.fit(X, y)
    return({'cv': fit, 'single': single, 'features': features})

def prepare_repeated(d, baseline, mw):
    # Select repeated measures and merge with baseline variables
    d = d[d['week'] <= mw].copy()
    d['col'] = d['variable'] + '_w' + d['week'].astype('str')
    r = d[['col', 'value']].pivot(columns='col', values='value')
    return(baseline.merge(r, left_index=True, right_index=True, how='inner'))

mrg = {'left_index': True, 'right_index': True, 'how': 'inner'}

if refit_iv_landscapes:
    cv_results = {}
    for k, v in samp.items():
        for max_week in [2, 4, 6]:
            # Prepare baseline features
            bl = baseline.loc[v].copy()
            if k[1] != 'both':
                bl.drop(labels=['escit'], axis=1, inplace=True)

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

            # Prepare outcome
            y = hdremit.loc[v].copy()

            # NOTE: We include baseline features in all models.

            # Option 1) Baseline only —————————————————————————————————————————
            print(k, max_week, '1. Baseline only')
            X = bl.copy()
            cv_results[("1. Baseline only",
                        k, max_week)] = evaluate_model(X, y, reps=n_reps)

            # Option 2) RM only ———————————————————————————————————————————————
            print(k, max_week, '2. RM only')
            X = bl.merge(rm, **mrg)
            cv_results[("2. RM only",
                        k, max_week)] = evaluate_model(X, y, reps=n_reps)

            # Option 3) RM + landscapes ———————————————————————————————————————
            print(k, max_week, '3. RM + LS')
            X = bl.merge(rm, **mrg).merge(ls, **mrg)
            cv_results[("3. RM + LS", k, max_week)] = evaluate_model(X, y, reps=n_reps)

            # Option 3: GC only ———————————————————————————————————————————————
            print(k, max_week, '4. GC only')
            X = bl.merge(rm, **mrg)
            # Run CV, incorporating growth curves
            cv_results[("4. GC only", 
                        k, max_week)] = evaluate_model(X, y,
                                                       reps=n_reps,
                                                       generate_curves=True)

            # Option 5: GC + landscapes ---------------------------------------
            print(k, max_week, '5. GC + LS')
            X = bl.merge(rm, **mrg).merge(ls, **mrg)
            X = bl.merge(rm, **mrg).merge(ls, **mrg)
            cv_results[("5. GC + LS",
                        k, max_week)] = evaluate_model(X, y, 
                                                       reps=n_reps,
                                                       generate_curves=True)
    dump(cv_results, filename=tstamp('cv_results'))

#  END
