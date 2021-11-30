# Title: Tune landscape parameters using scikit-learn
# Author: Ewan Carr
# Started: 2021-06-25

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
        RepeatedKFold, cross_validate)
from glmnet import LogitNet
import gudhi as gd
from gudhi.representations import DiagramSelector, Landscape
from scipy.stats import ttest_ind
from sklearn.metrics import (make_scorer, confusion_matrix,
                             recall_score, brier_score_loss)

refit_grid_search = False
refit_iv_landscapes = False
refit_iv_baseline = True
refit_iv_alts = True
select_subsample = False
n_reps = 50
cores = 10
grid_search = 'saved/2021_08_09/2021_08_08_153539_grid_search.joblib'

def tstamp(suffix):
    return(datetime.today().strftime('%Y_%m_%d_%H%M%S') + '_' + suffix + '.joblib')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                            Define functions                               ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

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

def cv_metric(l, reps=50):
    if reps == 1:
        return(np.mean(l))
    else:
        fold_means = [np.mean(i) for i in np.array_split(l, reps)]
        return(np.percentile(fold_means, [50, 2, 98]))

def print_summary(l, reps):
    for k1, v1 in l.items():
        print(k1)
        for k2, v2 in v1['cv'].items():
            if k2 != 'estimator':
                print(k2, cv_metric(v2, reps))

def reshape(dat):
    var = dat.columns.str.startswith('rep_')
    rv = dat.loc[:, var]. \
        melt(ignore_index=False)
    rv[['_', 'var', 'item', 'week']] = rv['variable']. \
        str.split('_', expand=True)
    rv['w'] = rv['week'].str.extract(r'(?P<week>\d+$)').astype('int')
    rv['var'] = rv['var'] + '_' + rv['item']
    rv.drop(labels=['_', 'week', 'variable', 'item'], axis=1, inplace=True)
    rv.sort_values(['subjectid', 'var', 'w'], inplace=True)
    return((rv, var))

def knn(dat):
    # kNN imputation, but retaining column names
    cols = dat.columns
    index = dat.index
    imp = KNNImputer(n_neighbors=5)
    dat = imp.fit_transform(dat)
    return(pd.DataFrame(dat, columns=cols, index=index))


def compute_topological_variables(dat,
                                  max_week=6,
                                  mas=1e5,
                                  fun='landscape',
                                  dims=[0, 1, 2],
                                  n_land=3,
                                  bins=10,
                                  keep_rm=False):

    # Select repeated measures and reshape from WIDE to LONG format
    rv, v = reshape(dat)

    # For each participant, generate landscape variables
    ls = {}
    for k in rv.index[~rv.index.duplicated()]:
        # Select this participant's rows
        # Reshape into grid of 'weeks' vs. 'measures'

        d = rv.loc[k, :]. \
            pivot(columns='var', values='value', index='w'). \
            loc[range(max_week + 1), :]. \
            values

        # Derive MDS components
        mds = MDS(n_components=3, random_state=42)
        d = mds.fit_transform(d.T)
        # Construct landscapes
        ac = gd.AlphaComplex(d)
        simplex_tree = ac.create_simplex_tree(max_alpha_square=mas)
        simplex_tree.compute_persistence()
        if fun == 'landscape':
            ps = {}
            # Construct landscapes in required dimensions
            for dim in dims:
                D = simplex_tree.persistence_intervals_in_dimension(dim)
                D = DiagramSelector(use=True,
                                    point_type="finite").fit_transform([D])
                if np.shape(D)[1] > 0:
                    LS = Landscape(num_landscapes=n_land, resolution=bins)
                    ps[dim] = LS.fit_transform(D)
                else:
                    ps[dim] = np.full((1, n_land*bins), 0)
            ls[k] = np.hstack([v for _, v in ps.items()])

    # Combine landscape variables for all participants
    ls = pd.DataFrame({k: v[0] for k, v in ls.items()}).T
    ls.set_axis(['X' + str(i) for i in ls.columns.values], axis=1, inplace=True)

    mrg = {'left_index': True, 'right_index': True, 'how': 'inner'}
    if keep_rm:
        # If we're keeping the repeated measures, first ensure we exclude
        # those measured beyond 'max_week'
        c = dat.columns[v]
        to_drop = np.array([])
        for w in [i for i in range(0, 27) if i > max_week]:
            to_drop = np.concatenate([to_drop,
                                      c[c.str.contains('_w' + str(w) + '$')]])
        rep = dat.drop(labels=to_drop, axis=1)
        # Merge landscapes with [baseline + repeated measures]
        X = rep.merge(ls, **mrg)
    else:
        # Merge landscapes with [baseline only]
        X = dat.loc[:, ~v].merge(ls, **mrg)
    return(X)

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

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃              Run CV for tuning parameters, without repetition             ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

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

def evaluate_model(X, y, reps=50, impute=False, return_estimator=False):
    features = X.columns
    if impute:
        X = knn(X)
    # Remove features with zero VarianceThreshold
    vt = VarianceThreshold()
    X = vt.fit_transform(X)
    # Run repeated CV
    clf = LogitNet(alpha=0.5,
                   cut_point=0,
                   n_splits=10,
                   random_state=42)
    rkf = RepeatedKFold(n_splits=10, n_repeats=reps, random_state=42)
    fit = cross_validate(clf,
                         X,
                         y,
                         cv=rkf,
                         scoring=scorers,
                         n_jobs=cores,
                         return_estimator=return_estimator)
    # Refit, once, to get feature importance
    single = clf.fit(X, y)
    return({'cv': fit, 'single': single, 'features': features})

if refit_iv_landscapes:
    cv_landscapes = {}
    for k, v in samp.items():
        for max_week in [2, 4, 6]:
            # For increasing weeks of data
            for keep_rm in [True, False]:
                print(k, max_week, keep_rm)
                # Prepare X/y
                X = comb.loc[v].copy()
                y = hdremit.loc[v].copy()
                if k[1] != 'both':
                    X.drop(labels=['escit'], axis=1, inplace=True)
                # Get best parameters from inner CV
                params = cv_inner[(k,
                                   keep_rm,
                                   max_week)].best_params_['topo__kw_args']
                params['keep_rm'] = keep_rm
                params['max_week'] = max_week
                # Impute missing values
                X = knn(X)
                # Generate landscape variables (once, rather than at each CV iteration)
                landscapes = compute_topological_variables(X, **params)
                # Remove features with zero VarianceThreshold
                vt = VarianceThreshold()
                feat = vt.fit_transform(landscapes)
                # Run repeated CV
                cv_landscapes[(k,
                               keep_rm,
                               max_week)] = evaluate_model(landscapes, y, reps=n_reps)
    # Save with date/time stamp
    dump(cv_landscapes, filename=tstamp('cv_landscapes'))
    del cv_landscapes

# Fit 'baseline only' model ---------------------------------------------------
if refit_iv_baseline:
    cv_baseline = {}
    for k, v in samp.items():
        print(k)
        # Prepare X/y
        X = baseline.loc[v].copy()
        y = hdremit.loc[v].copy()
        if k[1] != 'both':
            X.drop(labels=['escit'], axis=1, inplace=True)
        # Run repeated CV
        cv_baseline[k] = evaluate_model(X, y, n_reps, impute=True)
    # Save with date/time stamp
    dump(cv_baseline, filename=tstamp('cv_baseline'))
    del cv_baseline

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃           Fit models for growth curves / repeated measures only           ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Load growth curve parameters ------------------------------------------------
alts = {}
for mw in [2, 4, 6]:
    g = load('prediction/re_params/re_' + str(mw))
    g.columns = [i[0] + '_' + i[1].split('_')[1] for i in g.columns]
    g = baseline.merge(g, left_index=True, right_index=True, how='inner')
    alts['gc_' + str(mw)] = g

# Define sets of 'raw' repeated measures --------------------------------------
def reshape_rm(d, mw):
    d = d[d['week'] <= mw].copy()
    d['col'] = d['variable'] + '_w' + d['week'].astype('str')
    return(d[['col', 'value']].pivot(columns='col', values='value'))

for mw in [2, 4, 6]:
    r = reshape_rm(replong, mw)
    alts['rm_' + str(mw)] = baseline.merge(r,
            left_index=True,
            right_index=True,
            how='inner')

# Fit models for each ---------------------------------------------------------
if refit_iv_alts:
    cv_alts = {}
    for k1, id in samp.items():
        for k2, dat in alts.items():
            # Prepare X/y
            X = dat.loc[id].copy()
            y = hdremit.loc[id].copy()
            if k1[1] != 'both':
                X.drop(labels=['escit'], axis=1, inplace=True)
            # Run repeated CV
            cv_alts[(k1, k2)] = evaluate_model(X, y, reps=n_reps, impute=True)
    dump(cv_alts, filename=tstamp('cv_alts'))
