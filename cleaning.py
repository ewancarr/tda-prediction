# Clean repeated measures data from Sync.com folder
# 2021-01-29, 2021-02-17

from joblib import Parallel, delayed, dump
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.manifold import MDS
import gudhi as gd
from gudhi.representations import (
    DiagramSelector,
    Clamping,
    Landscape,
    Silhouette,
    BettiCurve,
    DiagramScaler)


# Functions -------------------------------------------------------------------

def process_repeated_measures(df, max_week=12):
    # Select columns
    df = df.drop('subjectid', axis=1).transpose()
    df.columns = ['value']
    df['variable'] = df.index
    df['week'] = df['variable'].str.extract(r'(\d+)$').astype('int64')
    df['measure'] = df['variable'].str.replace('\\d+$', '', regex=True)
    df.drop(['variable'], axis=1, inplace=True)
    # Reshape from LONG to WIDE
    df = df.pivot(index='week', values='value', columns='measure')
    # Remove weeks that are 100% missing across measures
    df = df.loc[df.isna().sum(axis=1) / len(df.columns) < 1, :]
    # Impute missing values, if any exist
    if df.isna().any().any():
        knn = KNNImputer(n_neighbors=5)
        dfi = knn.fit_transform(df)
    else:
        dfi = df.copy()
    # Scale
    sc = StandardScaler()
    scaled = pd.DataFrame(sc.fit_transform(dfi))
    # Relabel
    labels = df.columns[df.isna().sum() / len(df) < 1]
    scaled.columns = labels
    return(scaled)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                       Import repeated measures data                       ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# NOTE: We have two sources of repeated measures data:
#     1) the data used in the first draft of the review paper,
#        containing total scores only (2019);
#     2) the 2021 version, downloaded from Sync.com, which includes weekly
#        item-level scores.
# Here, I'm going to combine both sets.

# VERSION 1 (2019, TOTAL SCORES) ==============================================
gendep_core = pd.read_stata('data/GENDEP/raw/from_raquel/repeated_measures/' +
                            'gendep_core.dta')

# Select required measures
incl = 'subjectid|f[0-9][1-6]*score[0-9]+$|madrs[0-9]+$|hdrs[0-9]+$|bdi[0-9]+$'
rep19 = gendep_core.filter(regex=incl).set_index('subjectid')

# Reshape to LONG format
rep19 = rep19.melt(ignore_index=False)
rep19['week'] = rep19['variable'].str.extract('(\\d+)$')
rep19['variable'] = rep19['variable'].str.extract('(.*[a-z])\\d+$')

# VERSION 2 (2021, ITEM-LEVEL DATA) ===========================================
rep21 = pd.read_stata(Path('data/GENDEP/raw/from_raquel/files_from_sync')
                      / 'gendep-share-repeated-measures.dta')
incl = 'subjectid|week|hamd[0-9]+wk|madrs[0-9]+wk|bdi[0-9]+wk'
rep21 = rep21.filter(regex=incl).set_index('subjectid')

# Remove weeks before baseline
rep21 = rep21[rep21['week'] >= 0]

# Reshape to LONG format
rep21 = rep21.melt(id_vars='week', ignore_index=False)

# Rename variables (to make it clear that these are item-level measures)
rep21[['stub', 'item']] = rep21['variable'].str.extract('^([a-z]+)(\\d+)wk$')
rep21['variable'] = rep21['stub'] + '_q' + rep21['item']
rep21 = rep21[['variable', 'week', 'value']]
rep21['week'] = rep21['week'].astype('int32')

# Combine 'rep19' and 'rep21' -------------------------------------------------
replong = rep19.append(rep21)
replong['week'] = replong['week'].astype('int32')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                        Prepare baseline variables                         ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# NOTE: We have four sources of baseline data:

# 1. datClin.csv (randomised; n=430)
# 2. gendep_core.dta (n=868, loaded above)
# 3. data793.Rdata (larger, n=793)

# However, all measures from 'datClin.csv' are contained in 'gendep_core.dta',
# so we won't be using 'datClin.csv' here.

# gendep_core
req = ['age', 'sex', 'marital', 'educ', 'occup', 'children', 'ageonset',
       'drug', 'random', 'mscore', 'melanch', 'atscore', 'atyp', 'anxdep',
       'anxscore', 'anxsomdep', 'mhssri', 'everssri', 'mhtca', 'evertca',
       'mhdual', 'everdual', 'mhimao', 'everimao', 'mhoth', 'everoth',
       'mhantidep', 'everantidep', 'concssri', 'conctca', 'concantidep',
       'mcbenzo', 'mczhypn', 'mcestro', 'mhmirta', 'mcmirta', 'bmi_wk0',
       'f1score0', 'f2score0', 'f3score0', 'f61score0', 'f62score0',
       'f63score0', 'f64score0', 'f65score0', 'f66score0', 'madrs0', 'hdrs0',
       'bdi0']

baseline = gendep_core.set_index('subjectid').copy()[req]
baseline.columns = baseline.columns.str.replace('[_wk]*0$', '', regex=True)

# data793
data793 = pd.read_feather('data/GENDEP/clean/data793.feather')
data793.set_index('subjectid', inplace=True)
data793 = data793[['madrs1wk0', 'madrs2wk0', 'madrs3wk0', 'madrs4wk0',
                   'madrs5wk0', 'madrs6wk0', 'madrs7wk0', 'madrs8wk0',
                   'madrs9wk0', 'madrs10wk0', 'hamd1wk0', 'hamd2wk0',
                   'hamd3wk0', 'hamd4wk0', 'hamd5wk0', 'hamd6wk0', 'hamd7wk0',
                   'hamd8wk0', 'hamd9wk0', 'hamd10wk0', 'hamd11wk0',
                   'hamd12wk0', 'hamd13wk0', 'hamd14wk0', 'hamd15wk0',
                   'hamd16awk0', 'hamd16bwk0', 'hamd17wk0', 'bdi1wk0',
                   'bdi2wk0', 'bdi3wk0', 'bdi4wk0', 'bdi5wk0', 'bdi6wk0',
                   'bdi7wk0', 'bdi8wk0', 'bdi9wk0', 'bdi10wk0', 'bdi11wk0',
                   'bdi12wk0', 'bdi13wk0', 'bdi14wk0', 'bdi15wk0', 'bdi16wk0',
                   'bdi17wk0', 'bdi18wk0', 'bdi19awk0', 'bdi19wk0', 'bdi20wk0',
                   'bdi21wk0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8',
                   'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15', 'k16',
                   'k17', 'k18', 'k19', 'k20', 'k21', 'k22', 'k23', 'k24',
                   'k25', 'k26', 'k27', 'k28', 'k29', 'k30', 'k31', 'k32',
                   'k33', 'newbleq1wk0', 'newbleq2wk0', 'newbleq3wk0',
                   'newbleq4wk0', 'newbleq5wk0', 'newbleq6wk0', 'newbleq7wk0',
                   'newbleq8wk0', 'newbleq9wk0', 'newbleq10wk0',
                   'newbleq11wk0', 'newbleq12wk0', 'nevents', 'eventsbin']]

# Fix variable names
data793 = data793.melt(ignore_index=False)
data793['stub'] = data793['variable'].str.extract('([a-z]+)')
data793['item'] = data793['variable'].str.extract('[a-z]+(\\d*)')
data793['item'] = data793. \
    apply(lambda r: '_q' + r['item'] if r['item'] != '' else '', axis=1)
data793['measure'] = data793['stub'] + data793['item']
data793 = data793[['measure', 'value']].reset_index()
data793 = data793.pivot_table(index='subjectid',
                              columns='measure',
                              values='value')


# Check no overlapping variables
any(data793.columns.isin(baseline.columns))

# Append to other baseline variables
baseline = pd.concat([baseline, data793], axis=1, ignore_index=False)

# Check: are there any additional measures in the 'week 0' of repeated measures
# that we're missing from baseline?
for v in replong[replong['week'] == 0]['variable'].unique():
    if v not in baseline.columns:
        print('MISSING:         ', v)
    else:
        print('ALREADY PRESENT: ', v)

# Create dummy variables for categorical features
baseline['part01'] = np.select(
    [baseline['marital'] == 'Married/Cohab',
     baseline['marital'].isin(['Separated/Divorced',
                               'Widowed',
                               'Single'])],
    [1, 0])

baseline['occup01'] = np.select([baseline['occup'].isin(['unemployed',
                                                         'retired',
                                                         'student',
                                                         'homemaker']),
                                 baseline['occup'].isin(['full-time',
                                                         'part-time'])],
                                [1, 0])

baseline['nchild'] = np.select([baseline['children'] == 'No children',
                                baseline['children'] == '1 child',
                                baseline['children'] == '2 children',
                                baseline['children'] == '3+ children'],
                               [0, 1, 2, 3])

baseline['female'] = np.select([baseline['sex'] == 'male',
                                baseline['sex'] == 'female'],
                               [0, 1])

baseline.drop(labels=['occup', 'marital', 'children', 'sex'],
              axis=1, inplace=True)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                             Prepare outcomes                              ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

data793 = pd.read_feather('data/GENDEP/clean/data793.feather'). \
    set_index('subjectid')
outcomes = data793[['hdremit.all', 'mdpercadj']]
outcomes.columns = ['remit', 'perc']


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                       PREPARE TOPOLOGICAL VARIABLES                       ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# This takes the following steps:

# 1. Transpose the repeated measures, to maximise number of rows.
# 2. StandardScaler()
# 3. KNNImputer()
# 4. Extract MDS components
# 5. AlphaComplex()
# 6. create_simplex_tree()
# 7. Compute persistence
# 8. Construct landscape variables

# NOTE: we're not using Gower distance matrix, because we have no
# categorical repeated measures. We have ordinal items but these are
# treated as continuous by Gower.

def pow(n):
    return lambda x: np.power(x[1]-x[0], n)

# (a) running elastic net, first with the same approach re persistence
# landscapes in blue square (MDS, 3 dimensions, 12 landscapes, etc) (b) running
# elastic net, with the MDS approach, but only involving 1 dimension and 3
# landscapes (no worries if it's not easy to increase the number of points of
# discretisation, you already saw this fact was not affecting your results that
# much).


def describe_persistence(v,
                         fun='landscape',
                         mas=1e5,
                         bins=100,
                         n_land=12,
                         dims=[0, 1, 2]):
    if not set(dims).issubset([0, 1, 2]):
        raise ValueError('Requested dimensions must be in range [0, 1, 2].')
    # Impute missing values, round to integer
    knn = KNNImputer(n_neighbors=5)
    v = knn.fit_transform(v.T)
    # Extract MDS components
    mds = MDS(n_components=3)
    components = mds.fit_transform(v)
    # Construct Alpha complex
    ac = gd.AlphaComplex(points=components)
    # Construct Simplex tree, compute persistence
    simplex_tree = ac.create_simplex_tree(max_alpha_square=mas)
    simplex_tree.compute_persistence()
    # Compute requested persistence summary
    if fun == 'landscape':
        ps = {}
        for dim in range(3):
            D = simplex_tree.persistence_intervals_in_dimension(dim)
            D = DiagramSelector(use=True,
                                point_type="finite").fit_transform([D])
            if np.shape(D)[1] > 0:
                LS = Landscape(num_landscapes=n_land, resolution=bins)
                ps[dim] = LS.fit_transform(D)
            else:
                ps[dim] = np.full((1, bins*n_land), np.nan)
    elif fun == 'silouette':
        ps = {}
        for dim in range(3):
            D = simplex_tree.persistence_intervals_in_dimension(dim)
            if len(D) == 0:
                ps[dim] = np.full((1, bins), np.nan)
            else:
                D = DiagramSelector(use=True,
                                    point_type="finite").fit_transform([D])
                if np.shape(D)[1] > 1:
                    D = DiagramScaler(use=True,
                                      scalers=[([0, 1],
                                                MinMaxScaler())]). \
                        fit_transform(D)
                    D = DiagramScaler(use=True,
                                      scalers=[([1],
                                                Clamping(maximum=.9))]). \
                        fit_transform(D)
                    SH = Silhouette(resolution=bins, weight=pow(2))
                    ps[dim] = SH.fit_transform(D)
                else:
                    ps[dim] = np.full((1, bins), np.nan)
    elif fun == 'betti':
        ps = {}
        for dim in range(3):
            D = simplex_tree.persistence_intervals_in_dimension(dim)
            if len(D) == 0:
                ps[dim] = np.full((1, bins), 0)
            else:
                D = DiagramSelector(use=True,
                                    point_type="finite").fit_transform([D])
                if np.shape(D)[1] > 1:
                    D = DiagramScaler(use=True,
                                      scalers=[([0, 1],
                                                MinMaxScaler())]). \
                        fit_transform(D)
                    D = DiagramScaler(use=True,
                                      scalers=[([1],
                                                Clamping(maximum=.9))]). \
                        fit_transform(D)
                    BC = BettiCurve(resolution=bins)
                    ps[dim] = BC.fit_transform(D)
                else:
                    ps[dim] = np.full((1, bins), 0)
    # Keep requested dimensions
    selected_dimensions = {k: v for k, v in ps.items() if k in dims}
    return(np.hstack([v for k, v in selected_dimensions.items()]))


# Split repeated measures data into dictionary of participants ----------------
split = {}
for k, v in dict(tuple(replong.groupby('subjectid'))).items():
    split[k] = v.pivot(index='week',
                       columns='variable',
                       values='value').loc[0:4, :]

# Remove participants with 100% missing data ----------------------------------
for k in list(split):
    if split[k].isna().all().all():
        del split[k]

# Compute persistence summaries for each participant --------------------------

persistence = {}
for f, b in zip(['landscape', 'silouette', 'betti'], [500, 500, 500]):
    for lab, o in zip(['a', 'b'],
                      [{'dims': [0, 1, 2], 'n_land': 12},
                       {'dims': [0], 'n_land': 3}]):
        P = Parallel(n_jobs=24)(delayed(describe_persistence)(i,
                                                              fun=f,
                                                              bins=b,
                                                              **o)
                                for i in split.values())
        P = {k: v[0] for k, v in zip(split.keys(), P)}
        P = pd.DataFrame(P).T
        P.columns = ['X' + str(i) for i in list(P)]
        key = f + lab
        persistence[key] = P

for k, v in persistence.items():
    print(k, np.shape(v))

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃      Create WIDE version of repeated measures, for use in modelling       ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

repwide = replong.copy()
repwide['col'] = repwide['variable'] + '_w' + repwide['week'].astype('str')
repwide = repwide[['col', 'value']]
repwide = repwide.pivot(columns='col', values='value')

# Export everything -----------------------------------------------------------
inp = Path('analysis/prediction/GENDEP/v2/inputs')
dump(persistence, inp / 'topological_variables.joblib')
dump(outcomes, inp / 'outcomes.joblib')
dump(baseline, inp / 'baseline.joblib')
dump([replong, repwide], inp / 'repmea.joblib')

# Export to Feather -----------------------------------------------------------
repwide.reset_index().to_feather(inp / 'feather' / 'repwide.feather')
replong.reset_index().to_feather(inp / 'feather' / 'replong.feather')
