# Combine separate growth curve parameters into single data frame
# 2021-04-13

import os
from joblib import load, dump
import pandas as pd
from tqdm import tqdm, trange
import re
from pathlib import Path

# Load all growth curve parameters
index = load('growth_curves/index.joblib')
inputs = {}
for k, v in tqdm(index.items()):
    inputs[v] = load('growth_curves/outputs/' + str(k))

# Delete old versions
for f in Path('prediction/re_params/').glob('**/*'):
    f.unlink()

# Create versions for increasing numbers of weeks
vers = {}
for mw in trange(2, 13):
    vers[mw] = pd.concat({k: v for k, v in inputs.items() if int(re.findall(r'\d+', k)[-1]) == mw}, axis=1)
    dump(vers[mw], filename = 'prediction/re_params/re_' + str(mw))
