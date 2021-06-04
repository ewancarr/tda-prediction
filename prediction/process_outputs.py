import os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load, dump
from tqdm import tqdm, trange

def cv_metric(fit, reps=100):
    summary = {}
    for k, v in fit.items():
        fold_means = [np.mean(i) for i in np.array_split(v, reps)]
        summary[k] = np.percentile(fold_means, [50, 2, 98])
    return(summary)

sets_by, index = load('prediction/index.joblib')

# Load fitted models
n_fits = len(index)
missing = list()
stub = 'prediction/fits/'
fits = {}
for f in tqdm(index.keys()):
    fn = f + '.joblib'
    if os.path.exists(stub + fn):
        fits[fn] = {'file': fn,
                    'id': f,
                    'model': index[f],
                    'fit': load(stub + fn)}
    else:
        missing.append((f, index[f]))
print('Number missing:', len(missing))

# Split fits by estimator (random forest, elastic net)
by_estimator = {}
for k, v in fits.items():
    for crf, lab in zip(['random_forest', 'logit_net'], ['rf', 'en']):
        by_estimator[str(k) + '_' + lab] = {'file': v['file'],
                                           'id': v['id'],
                                           'model': v['model'],
                                           'fit': v['fit'][crf]}


for k, v in tqdm(by_estimator.items()):
    v['stat'] = cv_metric(v['fit'], reps=10)

summary_table = list()
for k, v in tqdm(by_estimator.items()):
    cell = {k2: f'{v2[0]:.3f} [{v2[1]:.3f}, {v2[2]:.3f}]' 
            for k2, v2 in v['stat'].items()}
    cell['model_id'], cell['model_type'] = k.split('_')
    cell['auc'], cell['auc_lo'], cell['auc_hi'] = v['stat']['test_auc']
    ret = pd.DataFrame(cell, index=[0])
    ret[['sample', 'drug', 'randomisation', 'id', 'hasbaseline', 'feat1', 'feat2']] = v['model'][0] + v['model'][1]
    summary_table.append(ret)

tab = pd.concat(summary_table).sort_values('auc', ascending=False)
# tab = tab[['id', 'model', 'test_auc', 'auc', 'auc_lo', 'auc_hi', 'sample', 'drug', 'randomisation', 'hasbaseline', 'feat1', 'feat2']]
# tab.sort_values(['sample', 'hasbaseline', 'model'], inplace=True)

tab.loc[(tab['feat1'] == 'reviewpaper') & (tab['feat2'].isnull())].to_csv('check.csv')
tab.loc[(tab['feat1'] == 'reviewpaper') & (tab['feat2'].isnull())]

tab.loc[tab['feat1'] == 'reviewpaper'][['model_id', 'model_type', 'id', 'auc', 'sample', 'randomisation', 'hasbaseline']]


tab.loc[tab['feat1'].isnull() & tab['feat2'].isnull()]
tab.loc[tab['feat1'] == 'reviewpaper']['model_id']

tab.to_csv('auc.csv', index=False)
tab.loc[not tab['feat2'], :]
tab.loc[(tab['hasbaseline'] == 'baseline') & (not tab['feat2'])]
tab[(tab['feat2'] == np.nan)]

tab['feat1'].value_counts()
tab['feat2'].value_counts()
type(tab['feat2'][0])
tab['feat2'][0].astype('int32')

