# Title:        Prepare tables and figures for TDA prediction paper
# Author:       Ewan Carr
# Started:      2021-06-22

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from joblib import load
import seaborn as sns
font = {'fontname': 'Arial'}
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

inp = Path('data')
outcomes = load(inp / 'outcomes.joblib')
baseline = load(inp / 'baseline.joblib')
replong, repwide = load(inp / 'repmea.joblib')
samp = load('samp.joblib')

def cv_metric(fit, reps=100):
    fold_means = [np.nanmean(i) 
            for i in np.array_split(fit['test_score'], reps)]
    return(np.percentile(fold_means, [50, 2, 98]))

# Load latest estimates from grid search
cv_landscapes = load('saved/2021_08_09/2021_08_11_082130_outer_cv.joblib')
_, cv_gc = load('saved/2021_08_09/2021_08_09_031950_outer_cv.joblib')

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
pooled['remit'] = pooled['remit'].astype('int32')                # type: ignore

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


table1 = pooled[req].groupby('samp').agg(calc_sum).T             # type: ignore
for r in table1.index:                                           # type: ignore
    table1.loc[r, 'label'] = labels[r][0]
    table1.loc[r, 'measure'] = labels[r][1]

# Get Ns 
c, a, b = pooled['samp'].value_counts()                          # type: ignore
table1.columns = [f'A (n={a})',                                  # type: ignore
                  f'B (n={b})',
                  f'C (n={c})',
                  'Label',
                  'Measure']
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


# Make figures ----------------------------------------------------------------

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
pal = sns.color_palette()
d['col'] = None
d['col'] = d['col'].astype(object)

for i, r in d.iterrows():
    if r['method'] == 'ls':
        d.at[i, 'nudge'] = -0.55
        d.at[i, 'col'] = pal[0]
        d.at[i, 'hatch'] = ''
    elif r['method'] == 'lsrm':
        d.at[i, 'nudge'] = -0.185
        d.at[i, 'col'] = pal[4]
        d.at[i, 'hatch'] = '//'
    elif r['method'] == 'gc':
        d.at[i, 'nudge'] = 0.185
        d.at[i, 'col'] = pal[2]
        d.at[i, 'hatch'] = ''
    elif r['method'] == 'rm':
        d.at[i, 'nudge'] = 0.55
        d.at[i, 'col'] = pal[3]
        d.at[i, 'hatch'] = ''
d['xpos'] = d['max_week'] + d['nudge']                           # type: ignore

subtitles = {'A': f'Escitalopram (n={str(a)})',
             'B': f'Nortriptyline (n={str(b)})',
             'C': f'Both drugs (n={str(c)})'}

fig, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(8, 10))
for ax, samp in zip(axes, ['A', 'B', 'C']):
    p = d.loc[d['sample'] == samp]
    ax.bar('xpos',
            'auc',
            data=p,
            width=0.30,
            color=p['col'],
            hatch=p['hatch'])
    ax.set_ylim(0.5, 1.0)
    errors = [p['hi'].values,
            p['lo'].values]
    yerr1 = p['auc'].values
    # Plot error bars
    yerr = [(p['auc'] - p['lo']).values,
            (p['hi'] - p['auc']).values]
    ax.errorbar(x=p['xpos'],
                y=p['auc'],
                yerr=yerr,
                fmt=' ',
                c='orange')
    for k, v in p.iterrows():
        ax.text(v['xpos'], v['hi'] + 0.02,
                f'{v["auc"]:.3f}',
                ha='center',
                size='x-small',
                **font)
        ax.text(v['xpos'],
                0.52,
                v['method'].upper(),
                ha='center', 
                size='x-small',
                c=lighten_color(v['col'], 0.3),
                **font)
    ax.set_title(subtitles[samp])
    ax.set_xticks([2, 4, 6])
    ax.tick_params(axis='x', bottom=False)
    if samp == 'A':
        methods = list()
        for lab, col in zip(['Landscapes',
                             'Landscapes and repeated measures',
                             'Growth curves',
                             'Repeated measures'],
                             [0, 4, 2, 3]):
            methods.append(mpatches.Patch(color=pal[col], label=lab))
        ax.legend(handles=methods,
                  ncol=2,
                  loc='upper left')
fig.supylabel('AUC')
fig.supxlabel('Number of weeks')
plt.tight_layout()
plt.savefig('figures/auc.png', dpi=600)

# Make table -----------------------------------------------------------------

auc_tab = d.pivot(index=['sample', 'drug', 'random', 'max_week'], 
                  columns='method',
                  values='cell') 
auc_tab = auc_tab[['ls', 'lsrm', 'gc', 'rm']]
auc_tab.to_excel('~/auc.xlsx')                                   # type: ignore

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                    Plot best AUC for each week of data                    ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

d['best'] = d.groupby(['sample', 'max_week'])['auc'].transform(max)
best = d[d['best'] == d['auc']][['sample', 'max_week', 'method',
    'auc', 'lo', 'hi']]. \
            sort_values(['sample', 'max_week'])
best['x'] = best['max_week'].astype('int')
best = pd.DataFrame({'method': ['rm', 'lsrm', 'gc'],
                     'label': ['Repeated measures',
                               'Repeated measures + Landscapes',
                               'Growth curves'],
                     'col': ['red', 'green', 'blue']}).merge(best, on='method')
best['method'] = best['method'].str.upper()
best['m'] = np.select([best['sample'] == 'A',
                       best['sample'] == 'B',
                       best['sample'] == 'C'],
                       ['o', '^', 's'])
best.sort_values(['sample', 'x'], inplace=True)

fig, ax = plt.subplots(figsize=(7, 5))
for label, df in best.groupby('sample'):
    for (x, y, method, c, m) in zip(df['x'], df['auc'], df['method'], df['col'], df['m']):
        ax.scatter(x=x, y=y, c=c, marker=m, label=None, s=45)
        ax.text(x, y + 0.005, method, ha='center', va='center', label=None)
    ax.plot('x', 'auc', data=df, color='gray', zorder=0, label=label) 
ax.set_xticks([2, 4, 6])
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_ylim(0.7, 0.9)
plt.tick_params(axis='x', bottom=False)
leg = [plt.Line2D([0], [0], marker='^', color='gray', label='Nortriptyline'),
       plt.Line2D([0], [0], marker='o', color='gray', label='Escitalopram'),
       plt.Line2D([0], [0], marker='s', color='gray', label='Combined')]
plt.legend(handles=leg)
plt.xlabel('Number of weeks')
plt.ylabel('AUC')
plt.tight_layout()
plt.savefig('figures/best.png', dpi=300)

def get_color(c):
    return (str(
        np.select([c == 'A', c == 'B', c == 'C'],
                  ['tab:blue', 'tab:green', 'tab:purple'])))


def get_linetype(b):
    return (str(np.select([b == 'baseline', b == 'nobaseline'],
                          ['o--', '.-'])))
