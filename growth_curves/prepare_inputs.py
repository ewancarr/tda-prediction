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

# inp = Path('inputs/growth_curves')
all_weeks, merged = load('data/merged.joblib')
outcomes = load('data/outcomes.joblib')

# Reshape repeated measures data from wide to long 

stubs = np.unique([i[:-4] 
           for i in all_weeks.drop(labels='subjectid', axis=1).columns])
dlong = pd.wide_to_long(all_weeks,
                        stubnames=stubs,
                        i='subjectid',
                        suffix='w[0-9]+',
                        sep='_',
                        j='t').reset_index()
dlong['week'] = [int(i[1:]) for i in dlong['t']]

# Create list containing all measures/weeks
m = dlong['week'].max()
opts = []
for max_weeks in range(2, m + 1):
    for label, content in dlong.drop(labels='t',
                                     axis=1).set_index(['subjectid',
                                                        'week']).items():
        values = content.copy().reset_index().dropna()
        # This is where we're setting the 'maximum number of weeks'
        # 0 = baseline, so 4 = baseline + weeks 1, 2, 3, 4 = 5 weeks total.
        values = values[values.week <= max_weeks]
        opts.append({'max_weeks': max_weeks,
                     'label': label,
                     'values': values})

# Save each option to disk
index = {}
for i, o in enumerate(opts):
    key = o['label'] + '_' + str(o['max_weeks'])
    index[i] = key
    dump(o, filename = 'growth_curves/inputs/' + str(i))

dump(index, filename = 'growth_curves/index.joblib')
