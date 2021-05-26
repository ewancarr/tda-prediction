# Title:        Prepare inputs for fitting growth curves on Rosalind
# Author:       Ewan Carr
# Started:      2021-04-13

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                      Prepare growth curve parameters                      ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from joblib import Parallel, delayed, dump, load
import multiprocessing
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import DistanceMetric
from sklearn.manifold import MDS
import gudhi as gd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

lf, repwide = load('data/repmea.joblib')
outcomes = load('data/outcomes.joblib')

# # Reshape repeated measures data from wide to long ----------------------------
# stubs = np.unique([i[0] for i in all_weeks.columns])
# lf = pd.melt(all_weeks, ignore_index=False)
# lf.columns = ['measure', 'week', 'value']

stubs = lf['variable'].unique()

# Create list containing all measures/weeks -----------------------------------
opts = []
for mw in range(2, 13):
    for s in stubs:
        val = lf[(lf.variable == s) &
                 (lf.week <= mw)].iloc[:, 1:3].reset_index()
        val.columns = ['subjectid', 'week', s]
        opts.append({'max_weeks': mw,
                     'label': s,
                     'values': val})

# Delete old versions ---------------------------------------------------------
for f in Path('growth_curves/inputs/').glob('**/*'):
    f.unlink()

# Save each option to disk ----------------------------------------------------
index = {}
for i, o in enumerate(opts):
    key = o['label'] + '_' + str(o['max_weeks'])
    index[i] = key
    dump(o, filename = 'growth_curves/inputs/' + str(i))

dump(index, filename = 'growth_curves/index.joblib')
