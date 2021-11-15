# Title:        Prepare tables and figures for TDA prediction paper
# Author:       Ewan Carr
# Started:      2021-06-22

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from joblib import load

inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
replong, repwide = load(inp / 'repmea.joblib')
samp = load('samp.joblib')

def cv_metric(fit, reps=100):
    fold_means = [np.nanmean(i) 
            for i in np.array_split(fit['test_score'], reps)]
    return(np.percentile(fold_means, [50, 2, 98]))


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                             List of features                              ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛




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

# Recoded several variables
pooled['remit'] = pooled['remit'].astype('int32')

def calc_sum(x):
    if x.dtype == 'float':
        return (f'{np.median(x):2.0f} ({x.min():2.0f}-{np.max(x):2.0f})')
    elif x.dtype in ['int', 'int32']:
        pct = 100 * x.mean()
        count = x.sum()
        return (f'{pct:2.0f}% (n={count})')

req = ['age', 'female', 'remit', 'occup01', 'samp']
labels = {'age': ('Age', 'Median [range]'),
          'female': ('Female gender', '% (n)'),
          'remit':   ('Remission', '% (n)'),
          'occup01': ('Employed', '% (n)')}


table1 = pooled[req].groupby('samp').agg(calc_sum).T
for r in table1.index:
    table1.loc[r, 'label'] = labels[r][0]
    table1.loc[r, 'measure'] = labels[r][1]

# Get Ns 
c, a, b = pooled['samp'].value_counts()
table1.columns = [f'A (n={a})', f'B (n={b})', f'C (n={c})', 'Label', 'Measure']
table1 = table1.iloc[:, [3, 4, 0, 1, 2]]

table1.to_excel('~/table1.xlsx')


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                       AUCs from internal validaton                        ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

cv_landscapes = load('saved/2021_08_09/2021_08_11_082130_outer_cv.joblib')
_, cv_gc = load('saved/2021_08_09/2021_08_09_031950_outer_cv.joblib')

# Landscape variables
cv = []
for k, v in cv_landscapes.items():
    p50, p2, p98 = cv_metric(v, reps=50)
    cell = f'{p50:.3f} [{p2:.3f}, {p98:.3f}]'
    cv.append({
        'sample': k[0][0],
        'drug': k[0][1],
        'random': k[0][2],
        'keep_rm': k[1],
        'max_week': int(k[2]),
        'auc': p50,
        'lo': p2,
        'hi': p98,
        'cell': cell})
cv = pd.DataFrame(cv)

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


# Make figures --------------------------------t--------------------------------

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


cv['method'] = np.where(cv['keep_rm'], 'lsrm', 'ls')
cv.drop(labels='keep_rm', axis=1, inplace=True)

d = pd.concat([gc, cv]).reset_index()

for i, r in d.iterrows():
    if r['method'] == 'ls':
        d.at[i, 'nudge'] = -0.55
        d.at[i, 'col'] = 'green'
        d.at[i, 'hash'] = ''
    elif r['method'] == 'lsrm':
        d.at[i, 'nudge'] = -0.185
        d.at[i, 'col'] = 'green'
        d.at[i, 'hash'] = '.'
    elif r['method'] == 'gc':
        d.at[i, 'nudge'] = 0.185
        d.at[i, 'col'] = 'red'
        d.at[i, 'hash'] = ''
    elif r['method'] == 'rm':
        d.at[i, 'nudge'] = 0.55
        d.at[i, 'col'] = 'blue'
        d.at[i, 'hash'] = ''
d['xpos'] = d['max_week'] + d['nudge']

subtitles = {'A': f'Escitalopram (n={str(a)})',
             'B': f'Nortriptyline (n={str(b)})',
             'C': f'Both drugs (n={str(c)})'}

fig, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(8, 10))
for ax, samp in zip(axes, ['A', 'B', 'C']):
    p = d.loc[d['sample'] == samp]
    ax.bar('xpos', 'auc', data=p, width=0.30, color=p['col'], hatch=p['hash'])
    ax.set_ylim(0.5, 1.0)
    errors = [p['hi'].values, p['lo'].values]
    yerr1 = p['auc'].values
    # Plot error bars
    yerr = [(p['auc'] - p['lo']).values, (p['hi'] - p['auc']).values]
    ax.errorbar(x=p['xpos'],
                y=p['auc'],
                yerr=yerr,
                fmt=' ',
                c='orange')
    for k, v in p.iterrows():
        ax.text(v['xpos'], v['hi'] + 0.02, f'{v["auc"]:.3f}', ha='center', size='x-small')
        ax.text(v['xpos'], 0.52, v['method'].upper(), ha='center', size='x-small', c=lighten_color(v['col'], 0.3))
    ax.set_title(subtitles[samp])
    ax.set_xticks([2, 4, 6])
    ax.tick_params(axis='x', bottom=False)
    if samp == 'A':
        ax.legend(handles=methods,
                   ncol=2,
                   loc='upper left')
methods = list()
for lab, col in zip(['Landscapes',
                     'Landscapes and repeated measures',
                     'Growth curves',
                     'Repeated measures'],
                     colors):
    methods.append(mpatches.Patch(color=col, label=lab))
fig.supylabel('AUC')
fig.supxlabel('Number of weeks')
plt.tight_layout()
plt.savefig('figures/auc.png', dpi=600)

# Make table -----------------------------------------------------------------

auc_tab = d.pivot(index=['sample', 'drug', 'random', 'max_week'], 
                  columns='method',
                  values='cell') 
auc_tab = auc_tab[['ls', 'lsrm', 'gc', 'rm']]
auc_tab.to_excel('~/auc.xlsx')


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                  Best performance for each week of data                   ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛



auc['week'] = auc.loc[:, 'feat1'].str.extract('[a-z_]+([0-9]+)')
auc['type'] = auc['feat1'].str.extract('([a-z]+_*[a-b]*)_*[0-9]+$')
weekly_max = auc[auc['hasbaseline'] == 'baseline'][['sample',
                                                    'type', 'week', 'auc',
                                                    'auc_lo', 'auc_hi',
                                                    'model_type']]. \
    dropna(subset=['type', 'week'], axis=0)
weekly_max['week'] = weekly_max['week'].astype('int32')
idx = weekly_max.groupby(['sample',
                          'week'])['auc'].transform(max) == weekly_max['auc']
weekly_max = weekly_max[idx]

weekly_max.pivot(index='sample', columns=['week'], values=['type'])
weekly_max.pivot(index='sample', columns=['week'], values=['model_type'])


def get_color(c):
    return (str(
        np.select([c == 'A', c == 'B', c == 'C'],
                  ['tab:blue', 'tab:green', 'tab:purple'])))


def get_linetype(b):
    return (str(np.select([b == 'baseline', b == 'nobaseline'],
                          ['o--', '.-'])))


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                                 AUC by s                                  ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150, sharey=True)
for d, l, i in zip([p_gc, p_rm], ['Growth curves', 'Repeated measures'],
                   [0, 1]):
    c = [get_color(i[0]) for i in d.columns]
    lt = [get_linetype(i[1]) for i in d.columns]
    d.plot.line(ax=axes[i], color=c, style=lt)
    axes[i].set_title(l)
fig.tight_layout()
