import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import gudhi as gd
from sklearn.metrics import (make_scorer, confusion_matrix,
                             recall_score, brier_score_loss,
                             accuracy_score,
                             balanced_accuracy_score)
from gudhi.representations import DiagramSelector, Landscape
from sklearn.base import BaseEstimator, TransformerMixin

#  ┌─────────────────────────────────────────────────┐ 
#  │                                                 │
#  │                Scoring functions                │
#  │                                                 │
#  └─────────────────────────────────────────────────┘

def fp(y_true, y_proba, threshold=0.5):
    y_pred = y_proba > threshold
    return(confusion_matrix(y_true, y_pred)[0, 1])


def fn(y_true, y_proba, threshold=0.5):
    y_pred = y_proba > threshold
    return(confusion_matrix(y_true, y_pred)[1, 0])


def tp(y_true, y_proba, threshold=0.5):
    y_pred = y_proba > threshold
    return(confusion_matrix(y_true, y_pred)[1, 1])


def tn(y_true, y_proba, threshold=0.5):
    y_pred = y_proba > threshold
    return(confusion_matrix(y_true, y_pred)[0, 0])


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
           'accbal': make_scorer(balanced_accuracy_score),
           'acc': make_scorer(accuracy_score),
           'tp': make_scorer(tp, needs_proba=True),
           'tn': make_scorer(tn, needs_proba=True),
           'fp': make_scorer(fp, needs_proba=True),
           'fn': make_scorer(fn, needs_proba=True),
           # TN
           'tn10': make_scorer(tn, needs_proba=True, threshold=0.1),
           'tn20': make_scorer(tn, needs_proba=True, threshold=0.2),
           'tn30': make_scorer(tn, needs_proba=True, threshold=0.3),
           'tn40': make_scorer(tn, needs_proba=True, threshold=0.4),
           'tn60': make_scorer(tn, needs_proba=True, threshold=0.6),
           'tn70': make_scorer(tn, needs_proba=True, threshold=0.7),
           'tn80': make_scorer(tn, needs_proba=True, threshold=0.8),
           'tn90': make_scorer(tn, needs_proba=True, threshold=0.9),
           # TP
           'tp10': make_scorer(tp, needs_proba=True, threshold=0.1),
           'tp20': make_scorer(tp, needs_proba=True, threshold=0.2),
           'tp30': make_scorer(tp, needs_proba=True, threshold=0.3),
           'tp40': make_scorer(tp, needs_proba=True, threshold=0.4),
           'tp60': make_scorer(tp, needs_proba=True, threshold=0.6),
           'tp70': make_scorer(tp, needs_proba=True, threshold=0.7),
           'tp80': make_scorer(tp, needs_proba=True, threshold=0.8),
           'tp90': make_scorer(tp, needs_proba=True, threshold=0.9),
           # FN
           'fn10': make_scorer(fn, needs_proba=True, threshold=0.1),
           'fn20': make_scorer(fn, needs_proba=True, threshold=0.2),
           'fn30': make_scorer(fn, needs_proba=True, threshold=0.3),
           'fn40': make_scorer(fn, needs_proba=True, threshold=0.4),
           'fn60': make_scorer(fn, needs_proba=True, threshold=0.6),
           'fn70': make_scorer(fn, needs_proba=True, threshold=0.7),
           'fn80': make_scorer(fn, needs_proba=True, threshold=0.8),
           'fn90': make_scorer(fn, needs_proba=True, threshold=0.9),
           # FP
           'fp10': make_scorer(fp, needs_proba=True, threshold=0.1),
           'fp20': make_scorer(fp, needs_proba=True, threshold=0.2),
           'fp30': make_scorer(fp, needs_proba=True, threshold=0.3),
           'fp40': make_scorer(fp, needs_proba=True, threshold=0.4),
           'fp60': make_scorer(fp, needs_proba=True, threshold=0.6),
           'fp70': make_scorer(fp, needs_proba=True, threshold=0.7),
           'fp80': make_scorer(fp, needs_proba=True, threshold=0.8),
           'fp90': make_scorer(fp, needs_proba=True, threshold=0.9),
           'brier': make_scorer(brier_score_loss,
                                greater_is_better=False,
                                needs_proba=True)}

#  ┌─────────────────────────────────────────────────────┐ 
#  │                                                     │
#  │                Topological functions                │
#  │                                                     │
#  └─────────────────────────────────────────────────────┘

def reshape(dat):
    var = dat.columns.str.startswith('rep__')
    rv = dat.loc[:, var].melt(ignore_index=False)
    rv[['_', 'var', 'week']] = rv['variable'].str.split('__', expand=True)
    rv['w'] = rv['week'].str.extract(r'(?P<week>\d+$)').astype('int')
    rv.drop(labels=['_', 'week', 'variable'], axis=1, inplace=True)
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
                loc[range(max_week + 1), :].values

        # Impute missing values (NOTE: This is done per-participant -- no data
        # leaking across participants). 
        imp = KNNImputer(n_neighbors=5)
        d = imp.fit_transform(d)

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


# Functions needed to genereate growth curves

class GenerateGrowthCurves(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X=None, y=None):
        return self
    def transform(self, X=None):
        return(fit_growth_curves(X))


