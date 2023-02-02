# Title:        Process results from separate inner/outer CV
# Author:       Ewan Carr
# Started:      2021-08-02

import numpy as np
import pandas as pd
from joblib import load

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

cv_landscapes = load('saved/2022-06-15 50 reps/2022_06_15_030446_cv_landscapes.joblib')
cv_gc = load('saved/2022-06-15 50 reps/2022_06_15_045105_cv_alts.joblib')

# Landscape variables
cv = {}
for k, v in cv_landscapes.items():
    # Get model information
    res['sample'] = k[0][0],
    res['drug'] = k[0][1],
    res['random'] = k[0][2],
    res['keep_rm' ] = k[1],
    res['max_week'] = int(k[2]),
    # Get CV metrics
    res = {}
    for met in [m for m  in list(v['cv']) if m.startswith('test_')]:
        p50, p2, p98 = cv_metric(v['cv'], met, reps=50)
        cell = f'{p50:.3f} [{p2:.3f}, {p98:.3f}]'
        res[met] = (p50, p2, p98, cell)
    cv[k] = res
    CONTINUE

cv[k]

cv.pivot(index=['sample', 'drug', 'random', 'max_week'], 
         columns=['keep_rm'],
         values='cell')

# Growth curves and repeated measures
gc = []
for k, v in cv_gc.items():
    p50, p2, p98 = cv_metric(v, reps=50)
    method, max_week = k[1].split('_')
    cell = f'{p50:.3f} [{p2:.3f}, {p98:.3f}]'
    gc.append({
        'sample': k[0][0],
        'drug': k[0][1],
        'random': k[0][2],
        'method': method,
        'max_week': int(max_week),
        'auc': p50,
        'lo': p2,
        'hi': p98,
        'cell': cell})
gc = pd.DataFrame(gc)

gc.pivot(index=['sample', 'drug', 'random', 'max_week'],
        columns=['method'],
        values=['cell'])

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                  Get results from 'baseline only' models                  ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


cv_baseline = load('saved/2021_11_18/2021_11_18_141553_cv_baseline.joblib')

bl = []
for k, v in cv_baseline.items():
    p50, p2, p98 = cv_metric(v, reps=100)
    cell = f'{p50:.3f} [{p2:.3f}, {p98:.3f}]'
    bl.append({
        'sample': k[0],
        'drug': k[1],
        'random': k[2],
        'max_week': 0,
        'auc': p50,
        'lo': p2,
        'hi': p98,
        'cell': cell})
bl = pd.DataFrame(bl)[['sample', 'drug', 'random', 'max_week', 'cell']].rename({'cell': 'bl'}, axis=1)
bl.set_index(['sample', 'drug', 'random', 'max_week'], inplace=True)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                       Table comparing all estimates                       ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Landscapes
tab_ls = cv.pivot(index=['sample', 'drug', 'random', 'max_week'], 
                  columns=['keep_rm'],
                  values='cell')
tab_ls.columns = ['ls', 'ls_rm']

# Growth curves, repeated measures
tab_gc = gc.pivot(index=['sample', 'drug', 'random', 'max_week'],
                  columns=['method'],
                  values=['cell'])
tab_gc.columns = ['gc', 'rm']

tab_bl = bl.pivot(index=['sample', 'drug', 'random', 'max_week'],
                  columns=['method'],
                  values=['cell'])
tab_bl.columns = ['bl', 'rm']

# Combined
how = {'left_index': True, 'right_index': True, 'how': 'outer'}
tab_all = pd.merge(tab_ls, tab_gc, **how)
tab_all.to_excel('~/results.xlsx')

