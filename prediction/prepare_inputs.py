# 2021-04-13

import numpy as np
import pandas as pd
from pathlib import Path

# This file loads data (prepared locally and uploaded to Rosalind) and
# prepares various 'configurations' of features/models for modelling.
# These are largely defined by the spreadsheet Raquel shared, reproduced below.

# NOTE: In addition to the below categories, we also want to run this for
# the escitalopram sample separately.

# | TDA component     | Type                | Predictors                         | Comments            |
# | ================= | =================== | ================================== | =================== |
# | NO TDA            | RAW                 | BASELINE                           |                     |
# | NO TDA            | RAW                 | BASELINE + WK1                     |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + WK2               |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK3         |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK4         |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK5         |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK6         |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK7         |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK8         |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK9         |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK10        |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK11        |                     |
# | NO TDA            | RAW                 | BASELINE + WK1 + ... + WK12        |                     |
# | NO TDA            | RAW                 | WK1 + ... + WK4                    |                     |
# | NO TDA            | GC                  | GC 2                               |                     |
# | NO TDA            | GC                  | GC 3                               |                     |
# | NO TDA            | GC                  | GC 4                               |                     |
# | NO TDA            | GC                  | GC 5                               |                     |
# | NO TDA            | GC                  | GC 6                               |                     |
# | NO TDA            | GC                  | GC 7                               |                     |
# | NO TDA            | GC                  | GC 8                               |                     |
# | NO TDA            | GC                  | GC 9                               |                     |
# | NO TDA            | GC                  | BASELINE + GC 2                    |                     |
# | NO TDA            | GC                  | BASELINE + GC 3                    |                     |
# | NO TDA            | GC                  | BASELINE + GC 4                    |                     |
# | NO TDA            | GC                  | BASELINE + GC 5                    |                     |
# | NO TDA            | GC                  | BASELINE + GC 6                    |                     |
# | NO TDA            | GC                  | BASELINE + GC 7                    |                     |
# | NO TDA            | GC                  | BASELINE + GC 8                    |                     |
# | NO TDA            | GC                  | BASELINE + GC 9                    |                     |
# | TDA               | LAND                | LAND                               |                     |
# | TDA               | LAND                | BASELINE + LAND                    |                     |
# | TDA               | BETTI               | BETTI                              |                     |
# | TDA               | BETTI               | BASELINE + BETTI                   |                     |
# | TDA               | SIL                 | SIL                                |                     |
# | TDA               | SIL                 | BASELINE + SIL                     |                     |
# | TDA               | GAUSS               | GAUSS                              | Not using.          |
# | TDA               | GAUSS               | BASELINE + GAUSS                   | Not using.          |
# | TDA               | COMBINED: RAW-LAND  | BASELINE + WK1 + ... + WK4 + LAND  |                     |
# | TDA               | COMBINED: RAW-LAND  | BASELINE + WK1 + ... + WK5 + LAND  |                     |
# | TDA               | COMBINED: RAW-LAND  | BASELINE + WK1 + ... + WK6 + LAND  |                     |
# | TDA               | COMBINED: RAW-BETTI | BASELINE + WK1 + ... + WK4 + BETTI |                     |
# | TDA               | COMBINED: RAW-BETTI | BASELINE + WK1 + ... + WK5 + BETTI |                     |
# | TDA               | COMBINED: RAW-BETTI | BASELINE + WK1 + ... + WK6 + BETTI |                     |
# | TDA               | COMBINED: RAW-SIL   | BASELINE + WK1 + ... + WK4 + SIL   |                     |
# | TDA               | COMBINED: RAW-SIL   | BASELINE + WK1 + ... + WK5 + SIL   |                     |
# | TDA               | COMBINED: RAW-SIL   | BASELINE + WK1 + ... + WK6 + SIL   |                     |
# | TDA               | COMBINED: GC-LAND   | SELECTED GC  +  LAND               |                     |
# | TDA               | COMBINED: GC-BETTI  | SELECTED GC  +  BETTI              |                     |
# | TDA               | COMBINED: GC-SIL    | SELECTED GC  +  SIL                |                     |

from joblib import load, dump
from pathlib import Path

# Load pre-prepared data ------------------------------------------------------
inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
drug = load(inp / 'drug.joblib')
X, X2 = load(inp / 'features.joblib')
repmea = load(inp / 'repmea.joblib')
res, gc_params = load(inp / 're.joblib')
all_weeks, merged = load(inp / 'merged.joblib')
landscapes, silouettes, betti = load(inp / 'landscapes.joblib')

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
sets['1_b1_NA'] = X         # Baseline features from datClin.csv
sets['2_b2_NA'] = X2       # Plus week 0 of repeated measures data
# Repeated measures only
i = 3
for k, v in rm.items():
    sets[str(i) + '_rm_' + str(k)] = v
    i += 1
# Repeated measures and baseline
for k, v in rm.items():
    for lab, bl in zip(['b1', 'b2'], [X, X2]):
        sets[str(i) + '_rm' + lab + '_' + str(k)] = v.merge(bl, **mrg)
        i += 1
# Growth curves only
for k, v in gc.items():
    sets[str(i) + '_gc_' + str(k)] = v
    i += 1
# Growth curves and baseline
for k, v in gc.items():
    for lab, bl in zip(['b1', 'b2'], [X, X2]):
        sets[str(i) + '_gc' + lab + '_' + str(k)] = v.merge(bl, **mrg)
        i += 1
# Topological variables
tda = {k: v for k, v in zip(['land', 'silo', 'betti'],
                            [landscapes, silouettes, betti])}
for k, v in tda.items():
    for lab, bl in zip(['b1', 'b2', 'NA'], [X, X2, None]):
        if lab == 'NA':
            sets[str(i) + '_' + k + '_NA'] = v
        else:
            sets[str(i) + '_' + k + lab + '_NA'] = v.merge(bl, **mrg)
        i += 1
# COMBINED: Topological variables PLUS [repeated measures OR growth curves]
for k, v in tda.items():
    for k2, v2 in rm.items():
        sets[str(i) + '_comb' + k + 'rm_' + str(k2)] = v.merge(v2, **mrg) 
        i += 1
    for k2, v2 in gc.items():
        sets[str(i) + '_comb' + k + 'gc_' + str(k2)] = v.merge(v2, **mrg) 
        i += 1

# Create two versions based on drug -------------------------------------------

# Option A: escitalopram only, no drug variable
# Option B: nordtrityline only, no drug variable
# Option C: both drugs, include 'drug'
pram = drug[drug['drug'] == 'escitalopram'].index
drug['escital'] = drug['drug'] == 'escitalopram'
by_drug = {}
for k, v in sets.items():
    by_drug['A_' + k] = v.loc[pram]
    by_drug['B_' + k] = v.drop(pram)
    by_drug['C_' + k] = drug[['escital']].merge(v, **mrg)

# Check keys are consistent
print(pd.DataFrame(pd.Series(by_drug.keys()).str.split('_')).to_string())

# Export list of models/features to Excel -------------------------------------
summary = {}
for k, v in by_drug.items():
    summary[k] = {'key': k,
                  'n_feat': np.shape(v)[1],
                  'n_participants': np.shape(v)[0],
                  'feat': ' '.join(v.columns.to_flat_index().str.join(''))}
pd.DataFrame(summary).T.to_csv('feature_sets.csv')

# Delete old versions ---------------------------------------------------------
for f in Path('prediction/sets/').glob('**/*'):
    f.unlink()

# Save all sets to disk ------------------------------------------------------
index = {}
for i, (k, v) in enumerate(by_drug.items()):
    dump(v, filename='prediction/sets/' + str(i))
    index[i] = k
dump(index, filename='prediction/index.joblib')
