# Fit a single growth curve model
# 2021-04-13
# Ewan Carr

import sys
from joblib import dump, load
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

def fit_gcm(p):
    # Fit a linear growth curve model with quadratic terms
    mdf = smf.mixedlm(p['label'] + ' ~ week + np.power(week, 2)', 
                      p['values'].dropna(), # NOTE: dropping missing rows
                      groups=p['values'].dropna()['subjectid'], 
                      re_formula='~ week + np.power(week, 2)')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mdf = mdf.fit()
        # Extract random effects
        re = pd.DataFrame(mdf.random_effects).T
        re.columns = [p['label'] + i for i in ['_int', '_t', '_t2']]
        return(re)

i = sys.argv[1]
opt = load('growth_curves/inputs/' + str(i))
params = fit_gcm(opt)
dump(params, filename = 'growth_curves/outputs/' + str(i))
