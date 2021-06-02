# Prepare feature sets for GENDEP prediction models
# Started: 2021-04-13
# Updated: 2021-05-26 

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load, dump
from tqdm import trange
import string
import re

# This file loads data (prepared locally and uploaded to Rosalind) and
# prepares various 'configurations' of features/models for modelling.

# Load pre-prepared data ------------------------------------------------------
inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
replong, repwide = load(inp / 'repmea.joblib')
persistence = load(inp / 'persistence.joblib')
landscapes_prev = load(inp /  'landscapes_prev.joblib')

# Format for keys, separated by '_':
# id    baseline    type1   type2

# Load growth curve parameters ------------------------------------------------
gc = {}
for mw in trange(2, 13, desc='Load growth curve parameters  '):
    gc[mw] = load('prediction/re_params/re_' + str(mw))
    gc[mw].columns = [i[0] + '_' + i[1].split('_')[1] for i in gc[mw].columns]

# Define sets of 'RAW' repeated measures --------------------------------------
def reshape_rm(d, mw):
    d = d[d['week'] <= mw].copy()
    d['col'] = d['variable'] + '_w' + d['week'].astype('str')
    return(d[['col', 'value']].pivot(columns='col', values='value'))

rm = {}
for mw in trange(2, 14, desc='Generate raw repeated measures'):
    rm[mw] = reshape_rm(replong, mw)

# --------------------------------------------------------------------------- #
# -------------------------- Define feature sets ---------------------------- #
# --------------------------------------------------------------------------- #

mrg = {'left_index': True,
       'right_index': True}
sets = {}

# Baseline only ---------------------------------------------------------------
bl = baseline.copy().drop(labels=['random', 'drug'], axis=1)
sets[(1, 'baseline', None, None)] = bl

# Repeated measures, with/without baseline ------------------------------------
i = 2
for k, v in rm.items():
    sets[(i, 'nobaseline', 'rm' + str(k), None)] = v
    i += 1
    sets[(i, 'baseline', 'rm' + str(k), None)] = v.merge(bl, **mrg)
    i += 1

# Growth curves, with/without baseline ----------------------------------------
for k, v in gc.items():
    sets[(i, 'nobaseline', 'gc' + str(k), None)] = v
    i += 1
    sets[(i, 'baseline', 'gc' + str(k), None)] = v.merge(bl, **mrg)
    i += 1

# Topological variables -------------------------------------------------------

# Add landscape variables from review paper (i.e. earlier approach)
persistence['reviewpaper'] = landscapes_prev

# Each set of topological variables, with/without baseline
for k, v in persistence.items():
    sets[(i, 'nobaseline', k, None)] = v
    i += 1
    sets[(i, 'baseline', k, None)] = v.merge(bl, **mrg)
    i += 1

# Combinations of topological variables plus GC, with/without baseline

# ----------------------------- NOT USING FOR NOW --------------------------- #
# # NOTE: for now, we're only looking at 2-6 weeks, to reduce number of models
# for k1, v1 in persistence.items():
#     if k1 != 'reviewpaper':
#         nw = int(re.findall(r'\d+', k1)[-1])
#         for k2, v2 in gc.items():
#             if nw <= 6 and k2 <= 6:
#                 sets[(i, 'nobaseline', k1, 'gc' + str(k2))] = v1.merge(v2, **mrg)
#                 i += 1
#                 sets[(i, 'baseline', k1, 'gc' + str(k2))] = v1.merge(v2, **mrg).merge(bl, **mrg)
#                 i += 1

# Select participants with complete outcome information -----------------------
has_outcome = outcomes['remit'].dropna().index
for k, v in sets.items():
    sets[k] = v.loc[v.index.intersection(has_outcome), :]

# Identify lowest sample size across all sets
lowest = 1e5
for k, v in sets.items():
    if len(v) < lowest:
        lowest = len(v)
        incl = v.index

# Select a consistent sample size across all sets
for k, v in sets.items():
    sets[k] = v.loc[v.index.intersection(incl), :]

for k, v in sets.items():
    print(len(v), k)

# Create multiple versions, based on randomisation/drug -----------------------

sets_by = {}
sets_by[('A', 'escitalopram', 'fullyrandom')] = baseline.loc[(baseline['random'] == 'randomized') & (baseline['drug'] == 'escitalopram'), ].index
sets_by[('B', 'nortriptyline', 'fullyrandom')] = baseline.loc[(baseline['random'] == 'randomized') & (baseline['drug'] == 'nortriptyline'), ].index
sets_by[('C', 'both', 'anyrandom')] = baseline.index

# Create binary measure of 'escitalopram'
drug = baseline[['drug']].copy()
drug.loc[:, 'escitalopram'] = drug['drug'] == 'escitalopram'
drug.drop(labels='drug', axis=1, inplace=True)
 
# Create versions of data for each sample (based on drug, randomisation)  
samples = {}
for k1, v1 in sets_by.items():
    for k2, v2 in sets.items():
        dat = v2.loc[v2.index.intersection(v1)].copy()
        if k1[1] == 'both':
            dat = dat.merge(drug, **mrg)
        samples[(k1, k2)] = dat

for k, v in samples.items():
    print(k, len(v))
        
# Save required sets to disk --------------------------------------------------

# Delete old versions
for f in Path('prediction/sets/').glob('**/*'):
    f.unlink()

# Save
index = {}
for i, (k, v) in enumerate(samples.items()):
    dump(v, filename='prediction/sets/' + str(i) + '.joblib')
    index[i] = k

# Save index
dump([sets_by, index], filename='prediction/index.joblib')

# # Export list of models/features to Excel -------------------------------------
# summary = {}
# for k, v in samples.items():
#     summary[k] = {'key': k,
#                   'sample': v['label'],
#                   'n_feat': np.shape(v['data'])[1],
#                   'n_participants': np.shape(v['data'])[0],
#                   'feat': ' '.join(v['data'].columns.to_flat_index().str.join(''))}
# pd.DataFrame(summary).T.to_csv('feature_sets.csv', index=False)
# dump(summary, 'feature_sets.joblib')

# Save copy of data to send to Raquel
letters = ['A', 'B', 'C']
i = 0
for k, v in samples.items():
    if k[1] == (1, 'baseline', None, None):
        print(i)
        v.to_stata('data/stata/' + letters[i] + '.dta')
        i += 1
