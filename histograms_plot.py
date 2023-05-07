#!/usr/bin/env python
# coding: utf-8

# # Histograms of metric values for selected IR & GR
# 
# ### imports, setup and loading data

# In[ ]:


import gc
import os
import warnings
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# In[ ]:


sample_size = 56
plots_dir = os.path.join('out', 'plots', f'n{sample_size}', 'histograms')
calculations_dir = os.path.join('out', 'calculations', f'n{sample_size}')

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(calculations_dir, exist_ok=True)

metrics = {
    'acc_equality_diff.bin': 'Accuracy equality',
    'equal_opp_diff.bin': 'Equal opportunity',
    'pred_equality_diff.bin': 'Predictive equality',
    'stat_parity.bin': 'Statistical parity',
    'neg_pred_parity_diff.bin': 'Negative predictive parity',
    'pos_pred_parity_diff.bin': 'Positive predictive parity',
}

plt.style.use('default')

# adjust font size on plots
SMALL_SIZE = MEDIUM_SIZE = 14
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# In[ ]:


# load IR & GR data for all confusion matrices of selected sample size
with open(path.join(calculations_dir, 'gr.bin'), 'rb') as f:
    gr = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['gr'])

with open(path.join(calculations_dir, 'ir.bin'), 'rb') as f:
    ir = pd.DataFrame(np.fromfile(f).astype(np.float16), columns=['ir'])


# ## Histograms with highlighted undefined values

# In[ ]:


def plot_histograms(metric_info, grs, irs, ratios_labels, bins_n):
    m_file, m_name = metric_info
    ir_labels = ratios_labels[::-1]
    gr_labels = ratios_labels

    with open(path.join(calculations_dir, m_file), 'rb') as f:
        df = pd.concat([gr, ir, pd.DataFrame(np.fromfile(f), columns=[m_name])], axis=1)

    # filter to get only results for selected ratios
    df = df.loc[df.ir.isin(irs) & df.gr.isin(grs)]

    print(f'{m_name}: inf: {df.loc[np.isinf(df[m_name])].shape[0]}, nan: {df.loc[np.isnan(df[m_name])].shape[0]}')

    # list like: [['a00', 'a00n', 'a01', 'a01n',...], ...]
    mosaic = [
        [f'a{i}{g}{x}'
         for g in range(len(grs))
         for x in ['', 'n']]
         for i in range(len(irs))
    ]

    fig, axs = plt.subplot_mosaic(mosaic,
                                  width_ratios=[50, 1]*len(grs),
                                  sharex=False, sharey=True,
                                  layout='constrained',
                                  # figsize=(18, 14),
                                  figsize=(20, 10),
                                  gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    # fig.suptitle(f'{m_name}: probabilities for selected IR & GR')
    fig.suptitle(f'{m_name}')

    for i, ir_val in enumerate(irs):
        for g, gr_val in enumerate(grs):

            # separate nans and numbers
            df_tmp = df.loc[(df.ir == ir_val) & (df.gr == gr_val)]
            total = df_tmp.shape[0]

            df_not_nan = df_tmp.loc[np.logical_not(np.isnan(df_tmp[m_name]))]
            nan_prob = df_tmp.loc[np.isnan(df_tmp[m_name])].shape[0] / total

            # prepare data for plotting
            binned, edges = np.histogram(df_not_nan[m_name], bins=bins_n)
            binned = binned / total

            # plot not nans
            axs[f'a{i}{g}'].hist(edges[:-1], edges, weights=binned, fc='black', ec='black')
            axs[f'a{i}{g}'].spines[['top', 'right']].set_visible(False)

            # plot nans - without drawing the full axis frame
            axs[f'a{i}{g}n'].bar(0, nan_prob, fc='red', ec='red', width=0.1, lw=0)
            axs[f'a{i}{g}n'].spines[['top', 'left']].set_visible(False)

            # styling
            if g == 0:
                axs[f'a{i}{g}'].set_ylabel(f'IR = {ir_labels[i]}')
            if i == 0:
                axs[f'a{i}{g}'].set_title(f'GR = {gr_labels[g]}')
            if i == len(irs) - 1:   # last row
                axs[f'a{i}{g}n'].set_xticks([0], ['Undef.'])
            else:
                axs[f'a{i}{g}'].set_xticklabels([])
                axs[f'a{i}{g}n'].set_xticks([0], [''])

    del df
    gc.collect()

    return fig


# In[ ]:


ratios = [1./28, 1./4, 1./2, 3./4, 27./28]
ratios_labels = ['1/28', '1/4', '1/2', '3/4', '27/28']

grs = np.float16(ratios)
irs = np.float16(ratios[::-1])

BINS = 109

for metric_info in metrics.items():
    fig = plot_histograms(metric_info, grs, irs, ratios_labels, BINS)
    fig.savefig(path.join(plots_dir, f'histogram_b{BINS}_{metric_info[1]}_titled.svg'), dpi=300)
    plt.close(fig)


# ### Histograms omitting undefined values
# 
# not used in the paper, but can be a useful reference

# In[ ]:


def plot_histograms_no_nan(metric_info, grs, irs, ratios_labels, bins_n):
    m_file, m_name = metric_info

    with open(path.join(calculations_dir, m_file), 'rb') as f:
        df = pd.concat([gr, ir, pd.DataFrame(np.fromfile(f).astype(np.float64), columns=[m_name])], axis=1)

    # filter to get only results for selected ratios
    df = df.loc[df.ir.isin(irs) & df.gr.isin(grs)]

    print(f'{m_name}: inf: {df.loc[np.isinf(df[m_name])].shape[0]}, nan: {df.loc[np.isnan(df[m_name])].shape[0]}')

    df = df.replace(np.inf, np.nan)

    fig, axs = plt.subplots(len(irs), len(grs),
                            sharey=True, sharex=True,
                            layout='constrained', figsize=(18, 14),
                            gridspec_kw={'wspace': 0.1,
                                         'hspace': 0.1})

    fig.suptitle(f'{m_name}: probabilities for selected IR & GR')

    for i, ir_val in enumerate(irs):
        for g, gr_val in enumerate(grs):

            # separate nans and numbers
            df_tmp = df.loc[(df.ir == ir_val) & (df.gr == gr_val)]
            total = df_tmp.shape[0]

            df_not_nan = df_tmp.loc[np.logical_not(np.isnan(df_tmp[m_name]))]

            # prepare data for plotting
            binned, edges = np.histogram(df_not_nan[m_name], bins=bins_n)
            binned = binned / total

            # plot not nans
            axs[i, g].hist(edges[:-1], edges, weights=binned, fc='black', ec='black')

            # styling
            # x-axis labels
            if g == 0:
                axs[i, g].set_ylabel(f'IR = {ratios_labels[i]}')

            # x-axis labels
            if i == 0:
                axs[i, g].set_title(f'GR = {ratios_labels[g]}')
            else:
                axs[i, g].set_xticklabels([])

    del df
    gc.collect()

    return fig


# In[ ]:


for metric_info in metrics.items():
    fig = plot_histograms_no_nan(metric_info, grs, irs, ratios_labels, BINS)
    fig.savefig(path.join(plots_dir, f'histogram_b{BINS}_{metric_info[1]}_no_nan.png'), dpi=300)
    # plt.show()
    plt.close(fig)


# ### histograms properties
# 
# quantitative properties of histograms for the selected metrics

# In[ ]:


def get_properties(metric_info, ratio_b, ratio_imb):
    m_file, m_name = metric_info
    print(f'\n\n===\n{m_name}:')

    with open(path.join(calculations_dir, m_file), 'rb') as f:
        df = pd.concat([gr, ir, pd.DataFrame(np.fromfile(f).astype(np.float64), columns=[m_name])], axis=1)

    df_b = df.loc[(df.ir == ratio_b) & (df.gr == ratio_b)]
    df_imb = df.loc[(df.ir == ratio_imb) & (df.gr == ratio_imb)]
    del df

    for df, ratio in [(df_b, ratio_b), (df_imb, ratio_imb)]:
        print(f' --> {ratio} <--')
        print(f'inf: {df.loc[np.isinf(df[m_name])].shape[0] / df.shape[0]}, nan: {df.loc[np.isnan(df[m_name])].shape[0] / df.shape[0]}')
        print(f'PPF: {df[df[m_name] == 0].shape[0] / df.shape[0]}')
        mode = df[m_name].mode()[0]
        print(f'Mode: {mode}\tprob: {df[df[m_name] == mode].shape[0] / df.shape[0]}')
        print(f'Unique vals: {df[m_name].nunique()}')


# In[ ]:


for metric_info in metrics.items():
    get_properties(metric_info, np.float16(1 / 2), np.float16(1 / 28))


# In[ ]:


def get_properties_selected(df, metric_info, gr, ir):
    m_file, m_name= metric_info
    print(f' --> IR: {ir}, GR: {gr} <--')

    if gr:
        cond_gr = df['gr'] == gr
    else:
        cond_gr = True
    if ir:
        cond_ir = df['ir'] == ir
    else:
        cond_ir = True

    if (cond_ir is not True) or (cond_gr is not True):
        df = df.loc[cond_gr & cond_ir]

    print(f'nan: {df.loc[np.isnan(df[m_name])].shape[0] / df.shape[0] * 100}%, inf: {df.loc[np.isinf(df[m_name])].shape[0] / df.shape[0] * 100}%')
    print(f'PPF: {df[df[m_name] == 0].shape[0] / df.shape[0] * 100}%')
    print(f'Unique vals: {df[m_name].nunique()}')


# In[ ]:


ratios = [float(2/56), float(1/2), float(54/56), None]

for metric_info in metrics.items():
    m_file, m_name= metric_info
    print(f'\n\n===\n{m_name}:')
    with open(path.join(calculations_dir, m_file), 'rb') as f:
        df = pd.concat([gr, ir, pd.DataFrame(np.fromfile(f).astype(np.float64), columns=[m_name])], axis=1)

    for g in ratios:
        for i in ratios:
            get_properties_selected(df, metric_info, g, i)
    del df

