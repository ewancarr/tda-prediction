# Title: Tune landscape parameters using scikit-learn
# Author: Ewan Carr
# Started: 2021-06-25

from pathlib import Path
from joblib import load, dump
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer 
from sklearn.impute import KNNImputer
from sklearn.manifold import MDS
from sklearn.model_selection import GridSearchCV
from glmnet import LogitNet
import gudhi as gd
from gudhi.representations import DiagramSelector, Landscape

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                            Define functions                               ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

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
                                  max_week=5,
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
            ls[k] = np.hstack([v for k, v in ps.items()])

    # Combine landscape variables for all participants
    ls = pd.DataFrame({k: v[0] for k, v in ls.items()}).T
    ls.set_axis(['X' + str(i) for i in ls.columns.values], axis=1, inplace=True)

    mrg = {'left_index': True, 'right_index': True, 'how': 'inner'}

    if keep_rm:
        # If we're keeping the repeated measures, first ensure we exclude
        # those measured beyond 'max_week'
        c = dat.columns[v]
        to_drop = np.array([])
        for w in [i for i in range(0, 26) if i > max_week]:
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
comb = comb.loc[incl, :]

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

for k, v in samp.items():
    print(k, 'n =', len(v))

# Recode 'drug'; remove 'random'
comb['escit'] = comb['drug'] == 'escitalopram'
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
        for mas in [10, 100, 1000, 2000, 1e5]:
            for dims in [[0], [0, 1], [0, 1, 2]]:
                for keep_rm in [True, False]:
                    for alpha in [[0.5], [1.0]]:
                        param_grid.append(
                                {'topo__kw_args': [
                                    {'fun': 'landscape',
                                     'n_land': land,
                                     'bins': bins,
                                     'dims': dims,
                                     'mas': mas,
                                     'keep_rm': keep_rm}],
                                 'estimator__alpha': alpha})

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃              Run CV for tuning parameters, without repetition             ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

fit = {}
for k, v in samp.items():
    X = comb.loc[v].copy()
    y = hdremit.loc[v].copy()
    if k[1] != 'both':
        X.drop(labels=['escit'], axis=1, inplace=True)
    # cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe,
                      param_grid,
                      cv=5,
                      scoring='roc_auc',
                      n_jobs=-1,
                      verbose=10)
    fit[k] = gs.fit(X, y)
    del X
    del y

dump(fit, filename='grid_search.joblib')

