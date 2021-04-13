# Combine separate growth curve parameters into single data frame
# 2021-04-13

import os
from joblib import load, dump

# Load all growth curve parameters
index = load('growth_curves/index.joblib')
inputs = {}
for k, v in tqdm(index.items()):
    inputs[v] = load('growth_curves/outputs/' + str(k))

# Create versions for increasing numbers of weeks
vers = {}
for mw in range(2, 15):
    vers[mw] = pd.concat({k: v for k, v in inputs.items() 
                          if int(k.split('_')[1]) == mw},
                         axis=1)
    dump(vers[mw], filename = 'prediction/inputs/re_' + str(mw))
