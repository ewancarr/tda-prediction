import os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load, dump
from tqdm import tqdm

def cv_metric(fit, reps=100):
    summary = {}
    for k, v in fit.items():
        fold_means = [np.mean(i) for i in np.array_split(v, reps)]
        summary[k] = np.percentile(fold_means, [50, 2, 98])
    return(summary)

sets_by, index = load('prediction/index.joblib')

# Load fitted models
n_fits = len(index)
n_miss = 0
stub = 'prediction/fits/'
fits = {}
for f in range(n_fits):
    if os.path.exists(stub + str(f)):
        print("Loaded: " + str(f) + " " + index[f])
        fits[f] = {'file': str(f),
                             'id': f,
                             'model': index[f],
                             'fit': load(stub + str(f))}
    else:
        # print("Not found: " + str(f) + " " + index[f])
        n_miss += 1
print('Number missing: ', n_miss)


# Split fits by estimator (random forest, elastic net)
by_estimator = {}
for k, v in fits.items():
    for crf, lab in zip(['random_forest', 'logit_net'], ['rf', 'en']):
        by_estimator[str(k) + '_' + lab] = {'file': v['file'],
                                           'id': v['id'],
                                           'model': v['model'],
                                           'fit': v['fit'][crf]}


for k, v in by_estimator.items():
    v['stat'] = cv_metric(v['fit'])

summary_table = []
for k, v in by_estimator.items():
    cell = {k2: f'{v2[0]:.3f} [{v2[1]:.3f}, {v2[2]:.3f}]' 
            for k2, v2 in v['stat'].items()}
    cell['model'] = k
    cell['label'] = v['model']
    cell['auc'] = v['stat']['test_auc'][0]
    cell['auc_lo'] = v['stat']['test_auc'][1]
    cell['auc_hi'] = v['stat']['test_auc'][2]
    summary_table.append(pd.DataFrame(cell,
                                      index=[0])[['model', 'label',
                                                  'test_auc', 'test_sens',
                                                  'test_spec', 'test_ppv',
                                                  'test_npv', 'test_brier',
                                                  'auc', 'auc_lo', 'auc_hi']])
summary_table = pd.concat(summary_table).sort_values('auc', ascending=False)

summary_table.sort_values('auc', ascending=False). \
        to_csv('auc.csv', index=False)
