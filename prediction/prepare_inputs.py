# Prepare feature sets for GENDEP prediction models
# Started: 2021-04-13
# Updated: 2021-04-29 

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load, dump
import string

# This file loads data (prepared locally and uploaded to Rosalind) and
# prepares various 'configurations' of features/models for modelling.
# These are largely defined by the spreadsheet Raquel shared.

# Load pre-prepared data ------------------------------------------------------
inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
repmea = load(inp / 'repmea.joblib')
# res, gc_params = load(inp / 're.joblib')
all_weeks, merged = load(inp / 'merged.joblib')
landscapes, silouettes, betti = load(inp / 'topological_variables.joblib')

# Load growth curve parameters ------------------------------------------------
gc = {}
for mw in range(2, 13):
    gc[mw] = load('prediction/inputs/re_' + str(mw))
    gc[mw].columns = [i[0] + '_' + i[1].split('_')[1] for i in gc[mw].columns]

# Define sets of 'RAW' repeated measures --------------------------------------
rm = {}
for mw in range(2, 14):
    rm[mw] = all_weeks.loc[:, (slice(None), range(0, mw + 1))]
    rm[mw].columns = [i + '_' + str(j) for i, j in rm[mw].columns]

# Define feature sets ---------------------------------------------------------

# NOTE: keys must follow a common format: id_label_weeks

mrg = {'left_index': True,
       'right_index': True}
sets = {}
# Baseline only
bl = baseline.drop(labels=['random', 'drug'], axis=1).copy()
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
# # Growth curves only
# for k, v in gc.items():
#     sets[str(i) + '_gc_' + str(k)] = v
#     i += 1
# # Growth curves and baseline
# for k, v in gc.items():
#     sets[str(i) + '_gcbl_' + str(k)] = v.merge(bl, **mrg)
#     i += 1
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
# # COMBINED: Topological variables PLUS [repeated measures OR growth curves]
# for k, v in tda.items():
#     for k2, v2 in rm.items():
#         sets[str(i) + '_comb' + k + 'rm_' + str(k2)] = v.merge(v2, **mrg) 
#         i += 1
#     for k2, v2 in gc.items():
#         sets[str(i) + '_comb' + k + 'gc_' + str(k2)] = v.merge(v2, **mrg) 
#         i += 1

# Create multiple versions, based on randomisation/drug -----------------------

sets_by = {}
letters = list(string.ascii_uppercase)
i = 0
for d in ['escitalopram', 'nortriptyline', 'both']:
    for r in ['randomized', 'non-random', 'both']:
        if (d == 'both') & (r == 'both'):
            pick = baseline.index
        elif (d == 'both'):
            pick = baseline[baseline['random'] == r].index
        elif (r == 'both'):
            pick = baseline[baseline['drug'] == d].index
        else:
            pick = baseline[(baseline['drug'] == d) &
                            (baseline['random'] == r)].index
        sets_by[letters[i]] = {'id': pick,
                               'label': (d, r)}
        i += 1

for k, v in sets_by.items():
    print(k, v['label'], len(v['id']))

samples = {}
for k1, v1 in sets_by.items():
    for k2, v2 in sets.items():
        dat = v2.loc[v2.index.intersection(v1['id'])].copy()
        if v1['label'][0] == 'both':
            dat.merge(baseline['drug'], left_index=True, right_index=True)
        samples[k1 + '_' + k2] = {'label': v1['label'],
                                  'data': dat}


# Drop participants with missing outcome information --------------------------

has_outcome = outcomes['remit'].dropna().index
for k, v in samples.items():
    v['data'] = v['data'].loc[v['data'].index.intersection(has_outcome)]
    print(k, v['label'], np.shape(v['data']))

# Export list of models/features to Excel -------------------------------------
summary = {}
for k, v in samples.items():
    summary[k] = {'key': k,
                  'sample': v['label'],
                  'n_feat': np.shape(v['data'])[1],
                  'n_participants': np.shape(v['data'])[0],
                  'feat': ' '.join(v['data'].columns.to_flat_index().str.join(''))}

pd.DataFrame(summary).T.to_csv('feature_sets.csv', index=False)

# Delete old versions ---------------------------------------------------------
for f in Path('prediction/sets/').glob('**/*'):
    f.unlink()

# Save all sets to disk ------------------------------------------------------
index = {}
for i, (k, v) in enumerate(samples.items()):
    dump(v, filename='prediction/sets/' + str(i))
    index[i] = k
dump(index, filename='prediction/index.joblib')
