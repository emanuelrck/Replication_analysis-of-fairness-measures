#!/usr/bin/env python
# coding: utf-8

# # Probability of Perfect Fairness and undefined values
#
# calculations for different metrics, group ratios and imbalance ratios

# In[ ]:


import os
import warnings
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import Timer

warnings.filterwarnings('ignore')
plt.style.use('default')

# adjust font size on plots
SMALL_SIZE = MEDIUM_SIZE = 16
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# In[ ]:


sample_size = 56
# setting epsilon to another (small non-negative) value allows to calculate the probability of being epsilon-close to perfect fairness
epsilon = 0

calculations_dir = path.join('out', 'calculations', f'n{sample_size}')
timer_dir = path.join('out', 'time')
os.makedirs(calculations_dir, exist_ok=True)
os.makedirs(timer_dir, exist_ok=True)
dataset_path = path.join('..', 'fairness-data-generator', 'out', f'Set(08,{sample_size}).bin')


# In[ ]:


## Calculate values for visualizations
diff_metrics = {  # { file: metric name }
    'pos_pred_parity_diff.bin': 'Positive predictive parity difference',
    'acc_equality_diff.bin': 'Accuracy equality difference',
    'stat_parity.bin': 'Statistical parity difference',
    'equal_opp_diff.bin': 'Equal opportunity difference',
    'neg_pred_parity_diff.bin': 'Negative predictive parity difference',
    'pred_equality_diff.bin': 'Predictive equality difference',
}


# In[ ]:


def calculate_ppf_diff(df, metrics, ratio_type, epsilon=0):
    pf_probs, nan_probs = {}, {}

    if epsilon == 0:
        compute_diff_prob = lambda df: np.sum(df['diff'] == 0) / len(df)
    else:
        compute_diff_prob = lambda df: np.sum(np.abs(df['diff']) < epsilon) / len(df)

    for metric_file, metric_name in metrics.items():
        with open(path.join(calculations_dir, metric_file), 'rb') as f:
            df = pd.concat([df, pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['diff'])], axis=1)

        pf_bygroup = list()
        nans_bygroup = list()

        for gn, group in df.groupby(ratio_type):
            if group['diff'].isna().all():
                pf_bygroup.append([gn, np.nan])
            else:
                pf_bygroup.append([gn, compute_diff_prob(group)])
            nans_bygroup.append([gn, group['diff'].isna().sum() / group.shape[0]])

        pf_bygroup = pd.DataFrame(pf_bygroup, columns=[ratio_type, 'diff'])
        pf_probs[metric_name] = pf_bygroup['diff']

        nans_bygroup = pd.DataFrame(nans_bygroup, columns=[ratio_type, 'diff'])
        nan_probs[metric_name] = nans_bygroup['diff']

        # the dataframe (first col) can be reused for the next metric
        df.drop('diff', axis=1, inplace=True)
        timer.checkpoint(f"calculate_ppf_diff {metric_name} ε={epsilon}")

    pf_probs[ratio_type] = pf_bygroup[ratio_type]
    pf_df = pd.DataFrame(pf_probs).reset_index()
    pf_df.to_csv(path.join(calculations_dir, f'perfect_fairness_{ratio_type}_eps{epsilon}.csv'), index=False)

    nan_probs[ratio_type] = nans_bygroup[ratio_type]
    nan_df = pd.DataFrame(nan_probs).reset_index()
    nan_df.to_csv(path.join(calculations_dir, f'nans_{ratio_type}.csv'), index=False)


# In[ ]:


timer = Timer().start()

for ratio in ['ir', 'gr']:
    print(ratio)
    try:
        with open(path.join(calculations_dir, f'{ratio}.bin'), 'rb') as f:
            df = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=[ratio])
        timer.checkpoint(f"load {ratio} file")
        calculate_ppf_diff(df, diff_metrics, ratio, epsilon)
    finally:
        del df

# del df
timer.reset()
timer.print()


# # Plotting
#
# reading data from the csv files created above

# In[ ]:


calculations_dir = path.join('out', 'calculations', f'n{sample_size}')
plots_dir = path.join('out', 'plots', f'n{sample_size}', 'perfect_fairness')
os.makedirs(plots_dir, exist_ok=True)

epsilons = [
    0,
]
ratio_types = ['gr', 'ir']


dfs = {
    (ratio_type, epsilon): pd.read_csv(path.join(calculations_dir, f'perfect_fairness_{ratio_type}_eps{epsilon}.csv'))
    for ratio_type in ratio_types
    for epsilon in epsilons
}

# colour scheme inspired by https://personal.sron.nl/~pault/
diff_metrics_styles = {
    'Accuracy equality difference': {'color': '#6699CC', 'marker': '*'},
    'Statistical parity difference': {'color': '#994455', 'marker': '.'},
    'Equal opportunity difference': {'color': '#004488', 'marker': 'v'},
    'Predictive equality difference': {'color': '#997700', 'marker': 'x'},
    'Negative predictive parity difference': {'color': '#EECC66', 'marker': '+'},
    'Positive predictive parity difference': {'color': '#EE99AA', 'marker': 'o'},
}

x_description = {
    'gr': 'Protected group ratio (GR)',
    'ir': 'Imbalance ratio (IR)',
}


# In[ ]:


def melt_df(df, base_metric):
    temp = df.pop(base_metric)
    length = len(df.columns)
    df = pd.melt(df)
    df[base_metric] = list(temp) * length
    return df


# In[ ]:


def plot_mlp(df, base_metric, color_mapping, title='Proportion of perfect fairness', y_max=None):
    fig, ax = plt.subplots(figsize=(9, 8))
    for col in color_mapping.keys():
        ax.plot(df[base_metric], df[col], label=col.replace('difference', ''), alpha=0.5, **color_mapping[col])

    if y_max is not None:
        ax.set_ylim(0, y_max)

    ax.set_xlabel(x_description[base_metric])
    ax.set_ylabel('Probability of perfect fairness')
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    return fig


# In[ ]:


timer.start()

for ratio_type in ratio_types:
    for eps in epsilons:
        fig = plot_mlp(
            dfs[(ratio_type, eps)], ratio_type, diff_metrics_styles, title='', y_max=1.0 if ratio_type == 'ir' else None
        )

        fig.savefig(path.join(plots_dir, f'ppf_{ratio_type}_zoom.pdf'), dpi=300)
        fig.savefig(path.join(plots_dir, f'ppf_{ratio_type}_square.svg'), dpi=300)
        timer.checkpoint(f"plot PPF {ratio_type} ε={eps}")

timer.reset()


# # Probability of NaN - plotting

# In[ ]:


def nan_probability(df, base_metric, color_mapping, title='Probability of NaN', y_max=None):
    fig, ax = plt.subplots(figsize=(9, 8))
    for col in color_mapping.keys():
        ax.plot(df[base_metric], df[col], label=col.replace('difference', ''), alpha=0.5, **color_mapping[col])

    if y_max is not None:
        ax.set_ylim(0, y_max)

    ax.set_xlabel(x_description[base_metric])
    ax.set_ylabel('Probability of undefined metric value')
    ax.spines[['top', 'right']].set_visible(False)

    ax.legend()
    fig.tight_layout()
    return fig


# In[ ]:


nan_dfs = {ratio_type: pd.read_csv(path.join(calculations_dir, f'nans_{ratio_type}.csv')) for ratio_type in ratio_types}

for ratio_type in ratio_types:
    fig = nan_probability(nan_dfs[ratio_type], ratio_type, diff_metrics_styles, title='', y_max=1.0)
    fig.savefig(path.join(plots_dir, f'nan_{ratio_type}_line.pdf'), dpi=300)
    fig.savefig(path.join(plots_dir, f'nan_{ratio_type}_square_line.svg'), dpi=300)

for ratio_type in ratio_types:
    fig = nan_probability(
        nan_dfs[ratio_type],
        ratio_type,
        diff_metrics_styles,
        title=f'Probability of NaN for given value of {ratio_type.upper()}',
        y_max=0.02,
    )
    fig.savefig(path.join(plots_dir, f'nan_{ratio_type}_square_zoom_line.pdf'), dpi=300)
    fig.savefig(path.join(plots_dir, f'nan_{ratio_type}_square_zoom_line.svg'), dpi=300)

timer.print()
timer.to_file(fn='ppf.csv')
