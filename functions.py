import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import gudhi as gd
from gudhi.representations import DiagramSelector, Landscape


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
                                  cluster='mds',
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
        if cluster == 'mds':
            mds = MDS(n_components=3, random_state=42)
            d = mds.fit_transform(d.T)
        elif cluster == 'pca':
            pca = PCA(n_components=3, random_state=42)
            d = pca.fit_transform(d.T)
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
    ls.columns = ['X' + str(i) for i in ls.columns]

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
