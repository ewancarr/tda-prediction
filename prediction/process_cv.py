# Title:        Process results from separate inner/outer CV
# Author:       Ewan Carr
# Started:      2021-08-02

import numpy as np
import pandas as pd
from joblib import load
from functions import *

def cv_metric(fit, what='test_score', reps=100):
    fold_means = [np.nanmean(i) 
            for i in np.array_split(fit[what], reps)]
    return(np.percentile(fold_means, [50, 2, 98]))

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                     Extract results from grid search                      ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

grid_search = load('saved/2021_08_09/2021_08_08_153539_grid_search.joblib')

best = []
for k, v in grid_search.items():
    best.append({
        'sample': k[0][0],
        'drug': k[0][1],
        'random': k[0][2],
        'keep_rm': k[1],
        'max_week': k[2],
        'auc': v.best_score_,
        'bins': v.best_params_['topo__kw_args']['bins'],
        'n_land': v.best_params_['topo__kw_args']['n_land'],
        'dims': v.best_params_['topo__kw_args']['dims'],
        'mas': v.best_params_['topo__kw_args']['mas']})
best = pd.DataFrame(best)

best.pivot_table(index=['sample', 'drug', 'random'], 
                 columns=['keep_rm', 'max_week'],
                 values=['auc']).round(3)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                 Extract results from internal validation                  ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

cv_results = load('saved/2023_02_22_055807_cv_results.joblib')
cv_results = load('saved/2023_03_22_203443_cv_results.joblib')


def make_cell(arr, reps=50):
    outer_folds = [np.nanmean(i) for i in np.array_split(arr, reps)]
    est = ['{:.3f}'.format(i) for i in np.percentile(outer_folds, [50, 2, 98])]
    return(f'{est[0]} [{est[1]}, {est[2]}]')

tab = []
for k, v in cv_results.items():
    row = {}
    row['model'] = k[0]
    row['sample'] = k[1][0]
    row['drug'] = k[1][1]
    row['random'] = k[1][2]
    row['max_week'] = k[2]
    for met in [m for m  in list(v['cv']) if m.startswith('test_')]:
        row[met] = make_cell(v['cv'][met], reps = 10)
    tab.append(row)

res = pd.DataFrame(tab).sort_values(['model', 'sample', 'max_week'])
res.to_excel('new_results.xlsx')

res[['model', 'sample', 'drug', 'max_week', 'test_auc']]. \
        pivot(index=['model', 'max_week'],
              columns=['sample', 'drug'],
              values='test_auc').round(3).to_excel('pivot.xlsx')

