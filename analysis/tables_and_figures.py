# Title:        Prepare tables and figures for TDA prediction paper
# Author:       Ewan Carr
# Started:      2021-06-22
# Updated:      2023-07-19

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import load
from functions import *
font = {'fontname': 'Arial'}
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Load datasets
inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
replong, repwide = load(inp / 'repmea.joblib')
samp = load('samp.joblib')

def cv_metric(arr, reps=100, squash=False):
    if reps == 0:
        fold_means = arr
    else:
        fold_means = [np.nanmean(i) for i in np.array_split(arr, reps)]
    p50, p2, p98 = np.percentile(fold_means, [50, 2.5, 97.5])
    if squash:
        return(f'{p50:.3f} [{p2:.3f}, {p98:.3f}]')
    else:
        return((p50, p2, p98))

for k in samp.keys():
    print(k)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                             List of features                              ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# This code checks that each variable used in our analysis is listed in 
# Supplementary Table 1, on Google Sheets.

check_variables(baseline, replong)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                                  Table 1                                  ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Create pooled dataset
pooled = []
for k, v in samp.items():
    d = baseline[baseline.index.isin(v)].copy()
    d['samp'] = k[0]
    pooled.append(d)
pooled = pd.concat(pooled)
pooled = pooled.merge(outcomes, left_index=True, right_index=True, how='left')

l1 = 'Mean (SD) [No. missing]'
l2 = '% (n) [No. missing]'
labels = {'age': ('Age¹', l1),
          'ageonset': ('Age at onset¹', l1),
          'female': ('Female gender²',  l2),
          'bmi': ('Body Mass Index (BMI)¹', l1),
          'madrs': ('Montgomery-Åsberg Depression Rating Scale (MADRS) score at baseline¹', l1),
          'hdrs': ('Hamilton Depression Rating Scale-17 item score at baseline¹', l1),
          'bdi': ('Beck Depression Inventory score at baseline¹', l1),
          'melanch': ('Melancholic subtype²', l2),
          'atyp': ('Atypical depression²', l2),
          'anxsomdep': ('Anxious-somatizing depression²', l2),
          'anxdep': ('Anxious depression²', l2),
          'eventsbin': ('Experiencing ≥ 1 Stressful Life Event during the 6 months before baseline²', l2),
          'everantidep': ('History of antidepressant treatment²', l2),
          'everssri': ('History of SSRI antidepressant²', l2),
          'evertca': ('History of Tricyclic antidepressant²', l2),
          'everdual': ('History of taking SNRI antidepressants²', l2),
          'remit':  ('Remission²', l2)}
req = list()
for k, v in labels.items():
    req.append(k)
    if v[1] == 'Mean (SD) [No. missing]':
        pooled[k] = pooled[k].astype(float)
    else:
        pooled[k] = pooled[k].fillna(-1).astype(int)
req.append('samp')

def calc_sum(x):
    # Count missing values
    x.where(x >= 0, np.nan, inplace=True)
    n_miss = x.isna().sum()
    # x.dropna(inplace=True)
    if x.dtype == 'float':
        return (f'{np.mean(x):2.1f} ({np.std(x):2.1f}) [{n_miss}]')
    elif x.dtype in ['int', 'int32']:
        pct = 100 * np.mean(x)
        count = np.sum(x)
        return (f'{pct:2.0f}% (n={count}) [{n_miss}]')

# NOTE: Haven't include 'smoker' because we didn't have this in the baseline
#       models.

table1 = pooled[req].groupby('samp').agg(calc_sum).T
for r in table1.index:
    table1.loc[r, 'label'] = labels[r][0]
    table1.loc[r, 'measure'] = labels[r][1]

# Get Ns
s_c, s_a, s_b = pooled['samp'].value_counts()
table1.columns = [f'A (n={s_a})',
                  f'B (n={s_b})',
                  f'C (n={s_c})',
                  'Label',
                  'Measure']
table1 = table1.iloc[:, [3, 4, 0, 1, 2]]
table1.to_excel('tables/table1.xlsx')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                     Check: data completeness by drug                      ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

for k, v in samp.items():
    print(k, repwide.loc[v].isna().sum(axis=1).median())

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                 Extract results from internal validation                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

cv = load('saved/final/2023_08_23_194541_cv_results.joblib')

n_reps = 20
for k1, v1 in cv.items():
    est = {}
    for k2, v2 in v1['cv'].items():
        est[k2] = cv_metric(v2, reps=n_reps, squash=False)
    est['test_spec30'] = cv_metric(specificity(v1['cv']['test_tn30'],
                                               v1['cv']['test_fp30']),
                                               reps=n_reps, squash=False)
    est['test_sens30'] = cv_metric(specificity(v1['cv']['test_tp30'],
                                               v1['cv']['test_fn30']),
                                               reps=n_reps, squash=False)
    est['test_spec40'] = cv_metric(specificity(v1['cv']['test_tn40'],
                                               v1['cv']['test_fp40']),
                                               reps=n_reps, squash=False)
    est['test_sens40'] = cv_metric(specificity(v1['cv']['test_tp40'],
                                               v1['cv']['test_fn40']),
                                               reps=n_reps, squash=False)
    v1['metrics'] = est

est = {}
for k, v in cv.items():
    est[k] = v['metrics']

pd.DataFrame(est).transpose().to_csv('metrics.csv')
