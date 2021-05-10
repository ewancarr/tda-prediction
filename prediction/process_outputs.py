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


set_by, index = load('prediction/index.joblib')

fits = {}
for f in Path('prediction/fits').iterdir():
    fits[int(f.stem)] = {'file': str(f),
                         'id': f.stem,
                         'model': index[int(f.stem)],
                         'fit': load(f)}

for k, v in fits.items():
    v['stat'] = cv_metric(v['fit'])

summary_table = []
for k, v in fits.items():
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
