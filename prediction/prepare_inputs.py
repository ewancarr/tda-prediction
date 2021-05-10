# Prepare feature sets for GENDEP prediction models
# Started: 2021-04-13
# Updated: 2021-04-29 

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load, dump
from tqdm import trange
import string

# This file loads data (prepared locally and uploaded to Rosalind) and
# prepares various 'configurations' of features/models for modelling.

# Load pre-prepared data ------------------------------------------------------
inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
replong, repwide = load(inp / 'repmea.joblib')
landscapes, silouettes, betti = load(inp / 'topological_variables.joblib')

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

# Define feature sets ---------------------------------------------------------

# NOTE: keys must follow a common format: id_label_weeks

mrg = {'left_index': True,
       'right_index': True}
sets = {}

# Baseline only
bl = baseline.copy().drop(labels=['random', 'drug'], axis=1)
sets['1_baseline'] = bl

# Repeated measures only
i = 2
for k, v in rm.items():
    sets[str(i) + '_rm_' + str(k)] = v
    i += 1

# Repeated measures and baseline
for k, v in rm.items():
    sets[str(i) + '_rmbl_' + str(k)] = v.merge(bl, **mrg)
    i += 1

# Growth curves only
for k, v in gc.items():
    sets[str(i) + '_gc_' + str(k)] = v
    i += 1

# Growth curves and baseline
for k, v in gc.items():
    sets[str(i) + '_gcbl_' + str(k)] = v.merge(bl, **mrg)
    i += 1

# Topological variables
tda = {k: v for k, v in zip(['land', 'silo', 'betti'],
                            [landscapes, silouettes, betti])}
for k, v in tda.items():
    for lab, dat in zip(['bl', 'NA'], [bl, None]):
        if lab == 'NA':
            sets[str(i) + '_' + k] = v
        else:
            sets[str(i) + '_' + k + lab] = v.merge(dat, **mrg)
        i += 1

# COMBINED: Topological variables PLUS [repeated measures OR growth curves]
for k, v in tda.items():
    for k2, v2 in rm.items():
        sets[str(i) + '_comb' + k + 'rm_' + str(k2)] = v.merge(v2, **mrg) 
        i += 1
    for k2, v2 in gc.items():
        sets[str(i) + '_comb' + k + 'gc_' + str(k2)] = v.merge(v2, **mrg) 
        i += 1

# Check that all feature sets have correct number of observations
for k, v in sets.items():
    if len(v) > 900:
        print(k)

# Create multiple versions, based on randomisation/drug -----------------------

sets_by = {}
letters = list(string.ascii_uppercase)
i = 0
for drug in ['escitalopram', 'nortriptyline', 'both']:
    for rand in ['randomized', 'non-random', 'both']:
        if (drug == 'both') & (rand == 'both'):
            pick = baseline.index
        elif (drug == 'both'):
            pick = baseline.loc[baseline['random'] == rand, ].index
        elif (rand == 'both'):
            pick = baseline.loc[baseline['drug'] == drug, ].index
        else:
            pick = baseline.loc[(baseline['drug'] == drug) &
                                (baseline['random'] == rand), ].index
        sets_by[letters[i]] = {'id': pick,
                               'label': (drug, rand)}
        i += 1

for k, v in sets_by.items():
    print(k, v['label'], len(v['id']))

# Create binary measure of 'escitalopram'
drug = baseline[['drug']].copy()
drug.loc[:, 'escitalopram'] = drug['drug'] == 'escitalopram'
drug.drop(labels='drug', axis=1, inplace=True)
 
samples = {}
for k1, v1 in sets_by.items():
    for k2, v2 in sets.items():
        dat = v2.loc[v2.index.intersection(v1['id'])].copy()
        if v1['label'][0] == 'both':
            dat = dat.merge(drug, left_index=True, right_index=True)
        samples[k1 + '_' + k2] = {'label': v1['label'],
                                  'data': dat}

# Check counts
mrg == {'left_index': True, 'right_index': True, 'how': 'left'}
check = baseline. \
        merge(drug, **mrg). \
        merge(outcomes, **mrg). \
        dropna(axis=0, subset=['remit'])
check.value_counts(subset=['drug', 'random'])


# Drop participants with missing outcome information --------------------------
has_outcome = outcomes['remit'].dropna().index
for k, v in samples.items():
    v['data'] = v['data'].loc[v['data'].index.intersection(has_outcome)]

# Check that all samples have 0/1 on outcome ----------------------------------
check = {}
for k, v in samples.items():
    y = v['data'].merge(outcomes, left_index=True, right_index=True)['remit']
    check[k] = [int(y.sum()), len(y)]
pd.DataFrame(check).T.to_csv('outcome_check.csv')

# Export list of models/features to Excel -------------------------------------
summary = {}
for k, v in samples.items():
    summary[k] = {'key': k,
                  'sample': v['label'],
                  'n_feat': np.shape(v['data'])[1],
                  'n_participants': np.shape(v['data'])[0],
                  'feat': ' '.join(v['data'].columns.to_flat_index().str.join(''))}
pd.DataFrame(summary).T.to_csv('feature_sets.csv', index=False)
dump(summary, 'feature_sets.joblib')

# Delete old versions ---------------------------------------------------------
for f in Path('prediction/sets/').glob('**/*'):
    f.unlink()

# Save all sets to disk ------------------------------------------------------
index = {}
for i, (k, v) in enumerate(samples.items()):
    dump(v, filename='prediction/sets/' + str(i))
    index[i] = k
dump([sets_by, index], filename='prediction/index.joblib')
