# Title:        Prepare tables and figures for TDA prediction paper
# Author:       Ewan Carr
# Started:      2021-06-22

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
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

def cv_metric(arr, reps=1000, squash=False):
    fold_means = [np.nanmean(i) for i in np.array_split(arr, reps)]
    p50, p2, p98 = np.percentile(fold_means, [50, 2, 98])
    if squash:
        return(f'{p50:.3f} [{p2:.3f}, {p98:.3f}]')
    else:
        return((p50, p2, p98))

# Load latest estimates from grid search
cv_landscapes = load('saved/2022-06-15 50 reps/2022_06_15_030446_cv_landscapes.joblib')
cv_baseline = load('saved/2022-06-15 50 reps/2022_06_15_031024_cv_baseline.joblib')
cv_alts = load('saved/2022-06-15 50 reps/2022_06_15_045105_cv_alts.joblib')

for k in samp.keys():
    print(k)

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                             List of features                              ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# This code checks that each variable used in our analysis is listed in 
# Supplementary Table 1, on Google Sheets.

def check_variables():
    url = 'https://docs.google.com/spreadsheets/d/1FApFC2HQJXNFdjYSQ7CLx7uss_eCsTEubsIMQTm9R_o/export?format=csv'
    lookup = pd.read_csv(url)

    # Check baseline variables ------------------------------------------------
    in_table = list(lookup['Variable*'].dropna())
    in_baseline = list(baseline)
    print('Variables in baseline dataset but not in Supplementary Table 1:')
    for v in list(baseline):
        if v not in in_table:
            print(v)

    print('Variables in Supplementary Table 1 that are not in baseline:')
    for v in in_table:
        if v not in list(baseline):
            print(v)

    # Check repeated measures variables ---------------------------------------
    in_repeated = replong['variable'].unique()
    table_repeated = lookup.dropna(subset=['Measured repeatedly'],
                                      axis='rows')[['Variable*',
                                                    'Measured repeatedly']]
    table_repeated = table_repeated[table_repeated['Measured repeatedly'] == 'Yes']['Variable*'].values

    print('Repeated measures variables that are not in Supplementary Table 1:')
    for v in list(in_repeated):
        if v not in table_repeated:
            print(v)

    print('Variables in Supplementary Table 1 that are not in the repeated measures dataset:')
    for v in table_repeated:
        if v not in list(in_repeated):
            print(v)

    len(table_repeated)

if False:
    check_variables()

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

l1 = 'Mean (SD) [No. missing]'
l2 = '% (n) [No. missing]'
labels = {'age': ('Age¹', l1),
          'ageonset': ('Age at onset¹', l1),
          'female': ('Female gender²',  l2),
          'bmi': ('Body Mass Index (BMI)¹', l1),
          'madrs': ('Montgomery-Åsberg Depression Rating Scale (MADRS) score at baseline¹', l1),
          'hdrs': ('Hamilton Depression Rating Scale-17 item score at baseline¹', l1),
          'bdi': ('Beck Depression Inventory score at baseline¹', l1),
          'melanch': ('Melancholic subtype²', l2),
          'atyp': ('Atypical depression²', l2),
          'anxsomdep': ('Anxious-somatizing depression²', l2),
          'anxdep': ('Anxious depression²', l2),
          'eventsbin': ('Experiencing ≥ 1 Stressful Life Event during the 6 months before baseline²', l2),
          'everantidep': ('History of antidepressant treatment²', l2),
          'everssri': ('History of SSRI antidepressant²', l2),
          'evertca': ('History of Tricyclic antidepressant²', l2),
          'everdual': ('History of taking SNRI antidepressants²', l2),
          'remit':  ('Remission²', l2)}
req = list()
for k, v in labels.items():
    req.append(k)
    if v[1] == 'Mean (SD) [No. missing]':
        pooled[k] = pooled[k].astype(float)
    else:
        pooled[k] = pooled[k].fillna(-1).astype(int)
req.append('samp')

def calc_sum(x):
    # Count missing values
    x.where(x >= 0, np.nan, inplace=True)
    n_miss = x.isna().sum()
    # x.dropna(inplace=True)
    if x.dtype == 'float':
        return (f'{np.mean(x):2.1f} ({np.std(x):2.1f}) [{n_miss}]')
    elif x.dtype in ['int', 'int32']:
        pct = 100 * np.mean(x)
        count = np.sum(x)
        return (f'{pct:2.0f}% (n={count}) [{n_miss}]')

# NOTE: Haven't include 'smoker' because we didn't have this in the baseline
#       models.

table1 = pooled[req].groupby('samp').agg(calc_sum).T
for r in table1.index:
    table1.loc[r, 'label'] = labels[r][0]
    table1.loc[r, 'measure'] = labels[r][1]

# Get Ns
s_c, s_a, s_b = pooled['samp'].value_counts()
table1.columns = [f'A (n={s_a})',
                  f'B (n={s_b})',
                  f'C (n={s_c})',
                  'Label',
                  'Measure']
table1 = table1.iloc[:, [3, 4, 0, 1, 2]]
table1.to_excel('~/table1.xlsx')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                       AUCs from internal validaton                        ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Collate all CV estimates ----------------------------------------------------

all_cv = []
# Landscapes
for k, v in cv_landscapes.items():
    all_cv.append({'sample': k[0][0],
                   'method': f'landscapes',
                   'rm': k[1],
                   'weeks': k[2],
                   'cv': v['cv'],
                   'feat': v['features'],
                   'sing': v['single']})
# Growth curves and repeated measures
for k, v in cv_alts.items():
    method, wk = k[1].split('_')
    rm = 'rm_only' if method == 'rm' else 'gc_only'
    print(k, method, wk, rm)
    all_cv.append({'sample': k[0][0],
                   'method': method,
                   'rm': rm,
                   'weeks': wk,
                   'cv': v['cv'],
                   'feat': v['features'],
                   'sing': v['single']})
# Baseline
for k, v in cv_baseline.items():
    all_cv.append({'sample': k[0][0],
                   'method': 'bl',
                   'rm': 'baseline_only',
                   'weeks': 0,
                   'cv': v['cv'],
                   'feat': v['features'],
                   'sing': v['single']})

# Calculate metrics, 95% CIs --------------------------------------------------
def sensitivity(tp, fn):
    return(tp / (tp + fn))

def specificity(tn, fp):
    return(tn / (tn + fp))

n_reps = 50
for i in tqdm(all_cv):
    est = {}
    for k, v in i['cv'].items():
        if k != 'estimator':
            est[k] = cv_metric(v, reps=n_reps, squash=False)
    est['test_spec30'] = cv_metric(specificity(i['cv']['test_tn30'],
                                               i['cv']['test_fp30']),
                                               reps=n_reps, squash=False)
    est['test_sens30'] = cv_metric(specificity(i['cv']['test_tp30'],
                                               i['cv']['test_fn30']),
                                               reps=n_reps, squash=False)
    est['test_spec40'] = cv_metric(specificity(i['cv']['test_tn40'],
                                               i['cv']['test_fp40']),
                                               reps=n_reps, squash=False)
    est['test_sens40'] = cv_metric(specificity(i['cv']['test_tp40'],
                                               i['cv']['test_fn40']),
                                               reps=n_reps, squash=False)
    i['est'] = est



# Combine into single table 
tab = []
rq = ['method', 'sample', 'rm', 'weeks']
for i in all_cv:
    row = {}
    for r in rq:
        row[r] = i[r]
    tab.append(row | i['est'])
tab = pd.DataFrame(tab).sort_values(rq, axis='rows')
tab.to_excel('cv_results.xlsx')

# Calculate sensitivity and specificity at different thresholds ---------------

# NOTE: not using this, because better to derive sens/spec based on all
# repetitions, as above. 

# def by_threshold(row, threshold=10, what='sens'):
#     tp = row['test_tp' + str(threshold)]
#     tn = row['test_tn' + str(threshold)]
#     fp = row['test_fp' + str(threshold)]
#     fn = row['test_fn' + str(threshold)]
#     if what == 'sens':
#         return((sensitivity(tp[0], fn[0]),
#                 sensitivity(tp[1], fn[1]),
#                 sensitivity(tp[2], fn[2])))
#     else:
#         return((specificity(tn[0], fp[0]),
#                 specificity(tn[1], fp[1]),
#                 specificity(tn[2], fp[2])))

# for measure in ['sens', 'spec']:
#     for t in [10, 20, 30, 40, 60, 70, 80, 90]:
#         tab[measure + '_' + str(t)] = tab.apply(by_threshold,
#                                                 threshold=t,
#                                                 what=measure,
#                                                 axis=1)


tab[['method',
     'sample',
     'test_sens',
     'test_spec'] + list(tab.columns[tab.columns.str.startswith('spec_')])].\
    applymap(lambda i: i[0]).to_excel('summary.xlsx')

# Select thresholds for sensitivity/specificity -------------------------------

# A, Escitalopram = 40%
# B, Nortriptyline = 30%
# C, Combined = 40%

t40 = tab['sample'].isin(['A', 'C'])
t30 = tab['sample'] == 'B'
for v in ['spec', 'sens']:
    tab.loc[t40, 'test_' + v + 'b'] = tab.loc[t40, 'test_' + v + '40']
    tab.loc[t30, 'test_' + v + 'b'] = tab.loc[t30, 'test_' + v + '30']

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


feat_types = ['BL', 'GC', 'LS', 'LSGC', 'LSRM', 'RM']
tab['m'] = np.select([(tab['method'] == 'bl') & (tab['rm'] == 'baseline_only'),
                      (tab['method'] == 'gc') & (tab['rm'] == 'gc_only'),
                      (tab['method'] == 'landscapes') & (tab['rm'] == 'ls_only'),
                      (tab['method'] == 'landscapes') & (tab['rm'] == 'plus_gc'),
                      (tab['method'] == 'landscapes') & (tab['rm'] == 'plus_rm'),
                      (tab['method'] == 'rm') & (tab['rm'] == 'rm_only')],
                     feat_types)

tab.drop(labels=['method', 'rm', 'fit_time'], axis=1, inplace=True)

pal = sns.color_palette()
d = tab.copy()

# Set colors
d['col'] = None
d['col'] = d['col'].astype(object)
for m, n, c, h in zip(feat_types,
                      CONTINUE HERE -- NEED TO UPDATE TO USE ALL 'FEAT_TYPES'
                      [0.75, -0.55, -0.185, 0.185, 0.55],
                      [7, 0, 4, 2, 3],
                      ['', '', '//', '', '']):
    d.loc[d.m == m, 'col'] = c
    d.loc[d.m == m, 'nudge'] = n
    d.loc[d.m == m, 'hatch'] = h
d['xpos'] = d['weeks'].astype(int) + d['nudge']
d['color'] = [pal[i] for i in d['col']]

# Extract AUC and CIs
d['auc'] = [i[0] for i in d['test_auc']]
d['lo'] = [i[1] for i in d['test_auc']]
d['hi'] = [i[2] for i in d['test_auc']]

subtitles = {'A': f'Escitalopram (n={str(s_a)})',
             'B': f'Nortriptyline (n={str(s_b)})',
             'C': f'Both drugs (n={str(s_c)})'}

fig, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(8, 10))
for ax, samp in zip(axes, ['A', 'B', 'C']):
    print(samp)
    p = d.loc[d['sample'] == samp]
    ax.bar(p['xpos'],
           p['auc'],
           width=0.30,
           color=p['color'],
           hatch=p['hatch'])
    ax.set_ylim(0.5, 1.0)
    # Plot error bars
    yerr = [p['auc'] - p['lo'],
            p['hi'] - p['auc']]
    ax.errorbar(x=p['xpos'],
                y=p['auc'],
                yerr=yerr,
                fmt=' ',
                c='orange')
    for k, v in p.iterrows():
        ax.text(v['xpos'],
                0.52,
                v['m'],
                ha='center',
                size='x-small',
                c=lighten_color(v['color'], 0.3),
                **font)
    ax.set_title(subtitles[samp])
    ax.set_xticks([2, 4, 6])
    ax.tick_params(axis='x', bottom=False)
    if samp == 'A':
        methods = list()
        for lab, col in zip(['Baseline only',
                             'Landscapes',
                             'Landscapes and repeated measures',
                             'Growth curves',
                             'Repeated measures'],
                             [7, 0, 4, 2, 3]):
            methods.append(mpatches.Patch(color=pal[col], label=lab))
        ax.legend(handles=methods,
                  ncol=2,
                  loc='upper left')
fig.supylabel('AUC')
fig.supxlabel('Number of weeks')
plt.tight_layout()
plt.savefig('figures/auc.png', dpi=600)

# Make table -----------------------------------------------------------------

tab['weeks'] = tab['weeks'].astype(int)
tab['sample'] = tab['sample'].astype(str)

def make_cell(t):
    return(f'{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]')

by = ['sample', 'weeks', 'm']
selected_results = {}
for v in ['test_auc', 'test_accbal', 'test_npv']:
    selected_results[v] = tab[v].apply(make_cell)

by_method = pd.concat([tab[by],
                       pd.concat(selected_results, axis=1)], axis=1)

by_method['weeks'] = by_method['weeks'].astype(int)
by_method.sort_values(by, inplace=True)
by_method = by_method[by_method['m'] != 'LS']
by_method.set_index(by).to_excel('supplementary_table_3.xlsx')

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                    Plot best AUC for each week of data                    ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

picks = ['test_auc', 'test_accbal', 'test_npv', 'test_sensb', 'test_specb']
best = {}
for p in picks:
    bysamp = tab.copy().join(tab.groupby(['sample', 'weeks'])[p]. \
                             apply(lambda x: max(x, key=lambda i:i[0])),
                             on=['sample', 'weeks'], rsuffix='_best')
    best[p] = bysamp.loc[bysamp[p] == bysamp[p + '_best'],
                         ['sample', 'weeks', 'm', p + '_best']]. \
        sort_values(['weeks', 'sample'])

def convert_title(k):
    return(np.select([k == 'test_auc',
                      k == 'test_accbal',
                      k == 'test_npv',
                      k == 'test_sensb',
                      k == 'test_specb'],
                     ['AUC',
                      'Balanced accuracy',
                      'NPV',
                      'Sensitivity',
                      'Specificity']))

for k, v in best.items():
    v['mark'] = np.select([v['sample'] == 'A',
                           v['sample'] == 'B',
                           v['sample'] == 'C'],
                          ['o', '^', 's'])
    v['col'] = np.select([v['sample'] == 'A',
                          v['sample'] == 'B',
                          v['sample'] == 'C'],
                         ['tab:red', 'tab:green', 'tab:blue'])
    v['jitter'] = np.select([v['sample'] == 'A',
                             v['sample'] == 'B',
                             v['sample'] == 'C'],
                            [-0.1, 0, 0.1])

# Plot for AUC, balanced accuracy, NPV
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 11))
for (var, dat), ax in zip(best.items(), axes.flatten()):
    dat['weeks'] = dat['weeks'].astype(int)
    dat['value'] = [i[0] for i in dat[var + '_best']]
    dat['lo'] = [i[1] for i in dat[var + '_best']]
    dat['hi'] = [i[2] for i in dat[var + '_best']]
    for label, df in dat.groupby('sample'):
        for (x, y, lo, hi, method, mark, j, col) in zip(df['weeks'],
                                                        df['value'],
                                                        df['lo'],
                                                        df['hi'],
                                                        df['m'],
                                                        df['mark'],
                                                        df['jitter'],
                                                        df['col']):
            ax.scatter(x=x+j, y=y, marker=mark, label=None, s=45, c=col)
            ax.text(x+j, y + 0.01,
                    method,
                    ha='center',
                    va='center',
                    size=9,
                    label=None)
            ax.vlines(x=x+j, ymin=lo, ymax=hi, colors=col)
        df['xj'] = df['weeks'] + df['jitter']
        ax.plot('xj', 'value', data=df, color=col, zorder=0, label=label)
        ax.set_title(convert_title(var))
        leg = [plt.Line2D([0], [0], marker='^', color='tab:green', label='Nortriptyline'),
               plt.Line2D([0], [0], marker='o', color='tab:red', label='Escitalopram'),
               plt.Line2D([0], [0], marker='s', color='tab:blue', label='Combined')]
        ax.legend(handles=leg, loc='upper left')
    ax.set_xticks([0, 2, 4, 6])
    ax.set_ylim(0.5, 0.90)
    ax.tick_params(axis='x', bottom=False)
ax.set_xlabel('Number of weeks')
plt.tight_layout()
axes[-1, -1].axis('off')
plt.savefig('figures/best.png', dpi=300)

# Make table of 'best' results ------------------------------------------------
tb = {}
for k, v in best.items():
    tb[k] = v.drop(labels=['mark', 'col', 'jitter',
                                 'value', 'lo', 'hi'],
                         axis=1). \
        set_index(['sample', 'weeks']) 
    tb[k].iloc[:, 1] = tb[k].iloc[:, 1]. \
        apply(lambda x: f'{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]')
pd.concat(tb, axis=1).to_excel('~/supplementary_table_2.xlsx')


# Calculate percentage improvement
perc = best['test_auc'][['sample', 'weeks', 'm', 'value']]
perc = perc.iloc[:3, :][['sample', 'value']]. \
    rename({'value': 'baseline'}, axis=1). \
    merge(perc, how='right', on='sample')
perc['diff'] = perc['value'] - perc['baseline']
perc['improv'] = 100 * (perc['diff'] / perc['value'])
perc.set_index(['sample']).pivot(values=['improv', 'diff'], columns='weeks') 
perc.set_index(['sample']).pivot(values='improv', columns='weeks').round(1)


# Calculate response on MADRS by week 2 ---------------------------------------

# (This is based on email from Raquel dated 2022-03-29).

ss = replong[(replong.variable == 'madrs') & (replong.week < 3)]
ss = pd.pivot(ss, columns='week', values='value')
ss['resp_w2'] = (ss[0] - ss[2]) / ss[0]

# By samples used in the paper
for (letter, drug, rand), ids in samp.items():
    print(drug, len(ids), ss.loc[ids]['resp_w2'].median().round(2))

# By drug/randomisation
comb = baseline[['drug', 'random']].merge(ss, left_index=True, right_index=True)
comb.groupby(['drug', 'random']).agg(['mean', 'count'])
