import os
from os import path
import pickle
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.ticker import PercentFormatter
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, recall_score
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import csv
from complexity import Complexity
from utils import Timer
SEED = 1000

#_mod -> deixou de ser de 0.01 a 0.99 para 0.05 a 0.95
# morecases ja tem mod e usa 2300 em vez de 1100
# race ja tem a mod e o numero max de casos para a race
DIRETORIA_EXP = 'exp_adult_seed1000_race_099'
# [majority, minority]
#SENSIVEL_VALUES = ['Male', 'Female']
#SENSIVEL_NAME = 'sex' # tentar fazer para race
#SAMPLE_SIZE = 2300 #2300 para sexo original era 1100

SENSIVEL_VALUES = ['White','notwhite']
SENSIVEL_NAME = 'race'
SAMPLE_SIZE = 1400
# ver se funciona
TARGET_VALUES = [0, 1]
TARGET_NAME = 'event'



#ver se dataset esta binarizado na raça se nao, binarizar e tirar 3 seeds de raça


def split_data(df, n, gr, ir,split_seed):
    """
    :param df: original data
    :param n: final size of the sample
    :return: the sample
    """
    # set ratios of sex and income
    f0 = round(n * gr * (1 - ir))
    f1 = round(n * gr * ir)
    m0 = round(n * (1 - gr) * (1 - ir))
    m1 = round(n * (1 - gr) * ir)

    sample = pd.concat(
        [
            df[(df[SENSIVEL_NAME] == SENSIVEL_VALUES[1]) & (df[TARGET_NAME] == TARGET_VALUES[0])].sample(n=int(f0), random_state=split_seed),
            df[(df[SENSIVEL_NAME] == SENSIVEL_VALUES[1]) & (df[TARGET_NAME] == TARGET_VALUES[1])].sample(n=int(f1), random_state=split_seed),
            df[(df[SENSIVEL_NAME] == SENSIVEL_VALUES[0]) & (df[TARGET_NAME] == TARGET_VALUES[0])].sample(n=int(m0), random_state=split_seed),
            df[(df[SENSIVEL_NAME] == SENSIVEL_VALUES[0]) & (df[TARGET_NAME] == TARGET_VALUES[1])].sample(n=int(m1), random_state=split_seed),
        ]
    ).reset_index(drop=True)
    return sample

def preprocess(dataset):
    
    X_all = dataset[features]
    y_all = LabelEncoder().fit_transform(dataset[TARGET_NAME])

    # encode categorical features
    data_encoder = OrdinalEncoder().fit(X_all[categorical_fs])
    X_categorical = data_encoder.transform(X_all[categorical_fs])

  
    # finally, the features
    X_all = np.concatenate([X_all.drop(categorical_fs, axis=1), X_categorical], axis=1)

    return X_all, y_all

def calculate_fairness(clf, X, y, protected, group=1, cls=1):
    """
    :param protected: id/name of the protected attribute column
    :param group: id of the protected group
    :param cls: id of the positive class
    :return: dictionary of fairness metrics for the given classifier's results
    """
    y_pred = clf.predict(X)
    # columns: protected_value, y_true, y_pred
    labelled = np.concatenate([X[:, protected].reshape(-1, 1), y.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)

    # calculate confusion matrices
    cms = [None, None]

    # y true/pred for the protected group
    ys = labelled[labelled[:, 0] == group]
    cms[0] = confusion_matrix(ys[:, 1], ys[:, 2], labels=[0, 1])
    # ... and for the other (unprotected) group
    ys = labelled[labelled[:, 0] != group]
    cms[1] = confusion_matrix(ys[:, 1], ys[:, 2], labels=[0, 1])

    # mj = majority - unprotected
    # mr = minority - protected
    mr, mj = group, 1 - group
    pos, neg = cls, 1 - cls

    # labels for the confusion matrix items
    tn = (neg, neg)
    fp = (neg, pos)
    fn = (pos, neg)
    tp = (pos, pos)

    # calculate fairness metrics
    fairness = dict()

    # Accuracy Equality Difference
    fairness['Accuracy Equality Difference'] = (cms[mj].item(tp) + cms[mj].item(tn)) / cms[mj].sum() - (
        cms[mr].item(tp) + cms[mr].item(tn)
    ) / cms[mr].sum()

    # Equal Opportunity Difference: j_tpr - i_tpr
    try:
        fairness['Equal Opportunity Difference'] = cms[mj].item(tp) / (cms[mj].item(tp) + cms[mj].item(fn)) - cms[
            mr
        ].item(tp) / (cms[mr].item(tp) + cms[mr].item(fn))
    except ZeroDivisionError:
        fairness['Equal Opportunity Difference'] = np.nan

    # Predictive Equality Difference: j_fpr - i_fpr
    try:
        fairness['Predictive Equality Difference'] = cms[mj].item(fp) / (cms[mj].item(fp) + cms[mj].item(tn)) - cms[
            mr
        ].item(fp) / (cms[mr].item(fp) + cms[mr].item(tn))
    except ZeroDivisionError:
        fairness['Predictive Equality Difference'] = np.nan

    # Positive Predictive Parity Difference: j_ppv - i_ppv
    try:
        fairness['Positive Predictive Parity Difference'] = cms[mj].item(tp) / (
            cms[mj].item(tp) + cms[mj].item(fp)
        ) - cms[mr].item(tp) / (cms[mr].item(tp) + cms[mr].item(fp))
    except ZeroDivisionError:
        fairness['Positive Predictive Parity Difference'] = np.nan

    # Negative Predictive Parity Difference: j_npv - i_npv
    try:
        fairness['Negative Predictive Parity Difference'] = cms[mj].item(tn) / (
            cms[mj].item(tn) + cms[mj].item(fn)
        ) - cms[mr].item(tn) / (cms[mr].item(tn) + cms[mr].item(fn))
    except ZeroDivisionError:
        fairness['Negative Predictive Parity Difference'] = np.nan

    fairness['Statistical Parity Difference'] = (cms[mj].item(tp) + cms[mj].item(fp)) / cms[mj].sum() - (
        cms[mr].item(tp) + cms[mr].item(fp)
    ) / cms[mr].sum()

    return fairness

def plot_fairness_gb_metric(fairness, gr, ir):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title(f'Fairness metrics for different classifiers; GR = {gr}, IR = {ir}')
    ax.set_ylabel('Fairness metric value')

    metrics = [
        '\n'.join([' '.join(f.split(" ")[:2]), ' '.join(f.split(" ")[2:])])
        for f in fairness[list(fairness.keys())[0]].keys()
    ]
    xticks = np.arange(len(metrics))
    width = 1.0 / (len(fairness.keys()) + 2)

    for i, (clf, f) in enumerate(fairness.items()):
        ax.bar(xticks + i * width, f.values(), width, label=clf.replace('Classifier', ''), color=colours[i])

    ax.set_xticks(xticks + width * len(fairness.keys()) / 2, metrics, rotation=45)
    ax.legend(ncols=1)
    plt.tight_layout()
    return fig

def plot_fairness_gb_clf(fairness, gr, ir):
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.set_title(f'Fairness metrics for different classifiers; GR = {gr}, IR = {ir}')
    ax.set_ylabel('Fairness metric value')

    metrics = fairness[list(fairness.keys())[0]].keys()
    classifiers = [c.replace('Classifier', '') for c in fairness.keys()]
    xticks = np.arange(len(classifiers))
    width = 1.0 / (len(metrics) + 2)
    shift = np.arange(len(metrics)) * width

    for i, (clf, f) in enumerate(fairness.items()):
        ax.bar(i + shift, f.values(), width, color=colours[: len(metrics)])
    ax.set_xticks(xticks + width * len(metrics) / 2, classifiers)
    ax.legend(handles=[mpatches.Patch(color=c, label=m) for c, m in zip(colours, metrics)], ncol=1)
    plt.tight_layout()

    return fig

def plot_line(fairness: pd.DataFrame, metric: str, ratio_type: str, fill='std', ylim=(-0.5, 0.5)):
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.set_ylabel(metric)
    ax.set_xlabel(ratio_type.upper())

    metrics = fairness['metric'].unique()
    clfs = fairness['clf'].unique()
    ratios = sorted(fairness[ratio_type].unique())
    other_ratio = 'gr' if ratio_type == 'ir' else 'ir'
    mean, stdev, err = {}, {}, {}

    for r in ratios:
        for clf in clfs:
            subset = fairness[
                (fairness[ratio_type] == r)
                & (fairness['clf'] == clf)
                & (fairness[other_ratio] == 0.5)
                & (fairness['metric'] == metric)
                & fairness['value'].notna()
            ]
            mean[(r, clf)] = subset['value'].mean(skipna=True)
            stdev[(r, clf)] = subset['value'].std(skipna=True)
            err[(r, clf)] = scipy.stats.sem(subset['value'], nan_policy='omit')

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)

    for i, clf in enumerate(clfs):
        ax.plot(ratios, [mean[(r, clf)] for r in ratios], label=clf, color=colours[i], marker='o')
        if fill == 'err':
            ax.fill_between(
                ratios,
                [mean[(r, clf)] - err[(r, clf)] for r in ratios],
                [mean[(r, clf)] + err[(r, clf)] for r in ratios],
                alpha=0.15,
                color=colours[i],
            )
        elif fill == 'std':
            ax.fill_between(
                ratios,
                [mean[(r, clf)] - stdev[(r, clf)] for r in ratios],
                [mean[(r, clf)] + stdev[(r, clf)] for r in ratios],
                alpha=0.15,
                color=colours[i],
            )

    ax.legend(loc=9)
    ax.spines[['top', 'right']].set_visible(False)

    # workaround to keep the x tick labels readable
    ratios_ticks = [
        '0.01',
        '  \n0.02',
        '0.05',
        '0.1',
        '0.2',
        '0.3',
        '0.4',
        '0.5',
        '0.6',
        '0.7',
        '0.8',
        '0.9',
        '0.95',
        '0.98\n  ',
        '0.99',
    ]

    ax.set_xticks(ratios, ratios_ticks, rotation=90)
    ax.set_xlim(0, 1)
    if ylim:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    return fig

def plot_line_abs(fairness: pd.DataFrame, metric: str, ratio_type: str, fill='std', ylim=None):
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.set_ylabel(metric)
    ax.set_xlabel(ratio_type.upper())

    metrics = fairness['metric'].unique()
    clfs = fairness['clf'].unique()
    ratios = sorted(fairness[ratio_type].unique())
    other_ratio = 'gr' if ratio_type == 'ir' else 'ir'
    mean, stdev, err = {}, {}, {}

    for r in ratios:
        for clf in clfs:
            subset = fairness[
                (fairness[ratio_type] == r)
                & (fairness['clf'] == clf)
                & (fairness[other_ratio] == 0.5)
                & (fairness['metric'] == metric)
                & fairness['value'].notna()
            ]
            mean[(r, clf)] = subset['value'].abs().mean(skipna=True)
            stdev[(r, clf)] = subset['value'].abs().std(skipna=True)
            err[(r, clf)] = scipy.stats.sem(subset['value'].abs(), nan_policy='omit')

    for i, clf in enumerate(clfs):
        ax.plot(ratios, [mean[(r, clf)] for r in ratios], label=clf, color=colours[i], marker='o')
        if fill == 'err':
            ax.fill_between(
                ratios,
                [mean[(r, clf)] - err[(r, clf)] for r in ratios],
                [mean[(r, clf)] + err[(r, clf)] for r in ratios],
                alpha=0.15,
                color=colours[i],
            )
        elif fill == 'std':
            ax.fill_between(
                ratios,
                [mean[(r, clf)] - stdev[(r, clf)] for r in ratios],
                [mean[(r, clf)] + stdev[(r, clf)] for r in ratios],
                alpha=0.15,
                color=colours[i],
            )

    ax.legend(loc=9)
    ax.spines[['top', 'right']].set_visible(False)

    ax.set_xticks(ratios, ratios, rotation=90)
    ax.set_xlim(0, 1)
    if ylim:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    return fig

def plot_nan(fairness, ratio_type, clfs=None, metrics=None, ylim=None):
    if clfs is None:
        clfs = fairness['clf'].unique()
    if metrics is None:
        metrics = fairness['metric'].unique()
    ratios = sorted(fairness[ratio_type].unique())
    other_ratio = 'gr' if ratio_type == 'ir' else 'ir'

    fig, ax = plt.subplots(2, (len(metrics) - 1) // 2 + 1, sharex=True, sharey=True, figsize=(12, 9))

    for i, metric in enumerate(metrics):
        ax[i % 2, i // 2].set_title(metric)
        ax[i % 2, i // 2].set_ylabel('NaN probability')
        ax[i % 2, i // 2].set_xlabel(ratio_type.upper())
        ax[i % 2, i // 2].yaxis.set_major_formatter(PercentFormatter(1))
        ax[i % 2, i // 2].spines[['top', 'right']].set_visible(False)

        for j, clf in enumerate(clfs):
            subset = fairness[
                (fairness['clf'] == clf) & (fairness[other_ratio] == 0.5) & (fairness['metric'] == metric)
            ]
            counts = subset.groupby(ratio_type)['value'].apply(lambda x: x.isna().sum() / x.shape[0])
            ax[i % 2, i // 2].plot(ratios, counts, label=clf, color=colours[j], marker='o', alpha=0.6)

    if ylim:
        ax[0, 0].set_ylim(*ylim)
    else:
        ax[0, 0].set_ylim(0, ax[0, 0].get_ylim()[1] * 1.1)
    ax[0, 0].set_xlim(0, 1)
    ax[0, 0].legend(loc=0)
    fig.tight_layout()

    return fig

def plot_line_all(fairness: pd.DataFrame, metrics: list[str], ratio_type: str, fill='std', ylim=(-0.5, 0.5)):
    fig, axs = plt.subplots((len(metrics) - 1) // 2 + 1, 2, sharex=True, sharey=True, figsize=(12, 12))

    for i, metric in enumerate(metrics):
        axs[i // 2, i % 2].set_ylabel(metric.replace('Difference', ''))

        metrics = fairness['metric'].unique()
        clfs = fairness['clf'].unique()
        ratios = sorted(fairness[ratio_type].unique())
        other_ratio = 'gr' if ratio_type == 'ir' else 'ir'
        mean, stdev, err = {}, {}, {}

        for r in ratios:
            for clf in clfs:
                subset = fairness[
                    (fairness[ratio_type] == r)
                    & (fairness['clf'] == clf)
                    & (fairness[other_ratio] == 0.5)
                    & (fairness['metric'] == metric)
                    & fairness['value'].notna()
                ]
                mean[(r, clf)] = subset['value'].mean(skipna=True)
                stdev[(r, clf)] = subset['value'].std(skipna=True)
                err[(r, clf)] = scipy.stats.sem(subset['value'], nan_policy='omit')

        axs[i // 2, i % 2].axhline(0, color='black', linestyle='--', alpha=0.9, lw=1)

        for j, clf in enumerate(clfs):
            axs[i // 2, i % 2].plot(
                ratios, [mean[(r, clf)] for r in ratios], label=clf, color=colours[j], marker='o', lw=1, alpha=0.85
            )
            if fill == 'err':
                axs[i // 2, i % 2].fill_between(
                    ratios,
                    [mean[(r, clf)] - err[(r, clf)] for r in ratios],
                    [mean[(r, clf)] + err[(r, clf)] for r in ratios],
                    alpha=0.15,
                    color=colours[j],
                )
            elif fill == 'std':
                axs[i // 2, i % 2].fill_between(
                    ratios,
                    [mean[(r, clf)] - stdev[(r, clf)] for r in ratios],
                    [mean[(r, clf)] + stdev[(r, clf)] for r in ratios],
                    alpha=0.15,
                    color=colours[j],
                )

        ratios_ticks = [
            '0.01\n',
            '0.02',
            '0.05',
            '0.1',
            '0.2',
            '0.3',
            '0.4',
            '0.5',
            '0.6',
            '0.7',
            '0.8',
            '0.9',
            '0.95',
            '0.98',
            '\n0.99',
        ]

        axs[i // 2, i % 2].spines[['top', 'right']].set_visible(False)
        axs[i // 2, i % 2].set_xticks(ratios, ratios_ticks, rotation=90)
        axs[i // 2, i % 2].set_xlim(0, 1)
        if i // 2 == 2:
            axs[i // 2, i % 2].set_xlabel(x_labels[ratio_type])
        if ylim:
            axs[i // 2, i % 2].set_ylim(*ylim)

    axs[0, 0].legend(loc=1, ncols=3)
    plt.tight_layout()
    return fig

def boxplot(fairness_results_cv, lowerDiv, upperDiv):
    
    metrics=[
            'Accuracy Equality Difference',
            'Statistical Parity Difference',
            'Equal Opportunity Difference',
            'Predictive Equality Difference',
            'Positive Predictive Parity Difference',
            'Negative Predictive Parity Difference',
    ]
    ynames=[
            'Accuracy Equality',
            'Statistical Parity',
            'Equal Opportunity',
            'Predictive Equality',
            'Positive Predictive Parity',
            'Negative Predictive Parity',
    ]
    classifiers = ['RandomForest',
                    'DecisionTree',
                    'GaussianNB',
                    'LogisticRegression',
                    'KNeighbors',
                    'MLP',
    ]
    #fazer por cada classificador é descomentar
    for ratio_type, other_ratio in [['ir', 'gr'], ['gr', 'ir']]:
        #for classifier in classifiers:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
            #fig.suptitle(ratio_type+' '+classifier)
            fig.suptitle(ratio_type)
            for idx,metric in enumerate(metrics):
                row = idx // 2  # Linha do subplot
                col = idx % 2   # Coluna do subplot
                subsetLower = fairness_results_cv[
                        (fairness_results_cv['metric'] == metric)
                        & (fairness_results_cv[ratio_type] < lowerDiv)
                        & (fairness_results_cv[other_ratio] == 0.5)
                        #& (fairness_results_cv['clf'] == classifier)
                        ]
                subsetMed = fairness_results_cv[
                        (fairness_results_cv['metric'] == metric)
                        & (fairness_results_cv[ratio_type] >= lowerDiv)
                        & (fairness_results_cv[ratio_type] < upperDiv)
                        & (fairness_results_cv[other_ratio] == 0.5)
                        #& (fairness_results_cv['clf'] == classifier)
                        ]
                subsetHigh = fairness_results_cv[
                        (fairness_results_cv['metric'] == metric)
                        & (fairness_results_cv[ratio_type] >= upperDiv)
                        & (fairness_results_cv[other_ratio] == 0.5)
                        #& (fairness_results_cv['clf'] == classifier)
                        ]

                subsetLower.dropna(inplace=True)
                subsetMed.dropna(inplace=True)
                subsetHigh.dropna(inplace=True)
                plot_idx = i * 2 + col + 1
                # Criar boxplot
                axes[row, col].boxplot([subsetLower['value'],subsetMed['value'],subsetHigh['value']])
                axes[row, col].set_xticklabels(['[0.01,'+str(lowerDiv)+'[','['+str(lowerDiv)+','+str(upperDiv)+'[', '['+str(upperDiv)+',0.99['])
                axes[row, col].set_ylabel(ynames[idx])
                axes[row, col].tick_params(axis='y', size=5)
            
            
            save_path = os.path.join(DIRETORIA_EXP, 'boxplot_'+ratio_type+'.png')
            plt.savefig(save_path)      
    return 
def measures_complexity(X,y,gr,ir):
    
    meta = [0] * len(X[0])
    complexity = Complexity(X,y,meta,"default")
    metrics = [[gr,ir],
        complexity.F1(),
        complexity.F1v(),
        complexity.F2(),
        complexity.F3(),
        complexity.F4(),
        complexity.R_value(),
        complexity.D3_value(),
        complexity.CM(),
        complexity.kDN(),
        complexity.T1(),
        complexity.DBC(),
        complexity.N1(),
        complexity.N2(),
        complexity.N3(),
        complexity.N4(),
        complexity.SI(),
        complexity.LSC(),
        complexity.input_noise(),
        complexity.borderline(),
        complexity.deg_overlap(),
        complexity.ICSV(),
        complexity.NSG(),
        complexity.Clust(),
        complexity.ONB()
    ]
    return metrics


def write_complexity_to_file(data):
    # Converter todos os arrays numpy para listas
    for i in range(len(data[1])):
        if isinstance(data[1][i], np.ndarray):
            data[1][i] = data[1][i].tolist()

    # Caminho do arquivo CSV
    file_path = DIRETORIA_EXP+'\metricas_complexidade.csv'

    # Escrever os dados no arquivo CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for row in data:
            writer.writerow(row)
    return


if __name__ == '__main__':
    seeds_split_data = [1120, 2928, 2379, 2050, 1962, 230, 825, 1781, 476, 1243, 1187, 1105, 2391, 2779, 1337, 2210, 1964, 2362, 376, 1437, 723, 485, 2033, 2815, 839, 1864, 1618, 546, 2938, 2796, 1028, 2388, 653, 264, 2489, 2531, 1778, 28, 2929, 1874, 1614, 313, 177, 1669, 2435, 1331, 2700, 1495, 140, 457]
    #seeds = [1000, 2000, 2137]
    seeds = [2137]
    sizes_samples = [1000,2000,2800] #max possiveis
    #sizes_samples = [1000,1000,1000]
    values_sens = [['Male','Female'],['White','notWhite'],['adult','notAdult']]
    names_sens =['sex','race','age_cat']
    upperBound_boxplot = 0.8
    lowerBound_boxplot = 0.2
 
    for i in seeds:
        SEED = i
        for j in range(len(names_sens)):
            SENSIVEL_VALUES = values_sens[j]
            SENSIVEL_NAME = names_sens[j]
            SAMPLE_SIZE = sizes_samples[j]
            DIRETORIA_EXP = "compas__seed"+str(SEED)+"_"+str(SENSIVEL_NAME)+"_size"+str(SAMPLE_SIZE)
            

            warnings.filterwarnings('ignore')
            plt.style.use('default')

            # colour scheme inspired by https://personal.sron.nl/~pault/
            colours = ['#EE7733', '#33BBEE', '#EE3377', '#888888', '#009988', "#332288"]

            x_labels = {
                'gr': 'Protected group ratio (GR)',
                'ir': 'Imbalance ratio (IR)',
            }

            SMALL_SIZE = 10
            MEDIUM_SIZE = 13
            BIGGER_SIZE = 15

            plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
            plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        
            timer_dir = path.join(DIRETORIA_EXP, 'time')

            #-------------------------- Tratar dataset

            timer = Timer().start()

            features = [
                "sex",
                "age_cat",
                "race",
                "juv_fel_count",
                "juv_misd_count",
                "juv_other_count",
                "priors_count",
                "c_charge_degree",
                "score_text",
                "v_score_text",
            ]
            dataset = pd.read_csv('data/compasTransformed.data', sep=';', na_values=['?', ' ?'], header=0, names=features + [TARGET_NAME])
            # Verificar a transformação
            #print(dataset['race'].value_counts())
            # ------------------------------------------Transformar o atributo 'race'
            
            # Verificar a transformação
            #print(dataset['race'].value_counts())
        

            timer.checkpoint(f"read data")
            timer.reset()

            plots_dir = os.path.join(DIRETORIA_EXP, 'plots', 'case_study', 'census_income')
            os.makedirs(plots_dir, exist_ok=True)

            # ### splitting the data into subsets
            #
            # To enable comparison for different GR and IR.
            #
            # The split is proportional to group/class sizes
            n = dataset.shape[0]
            gr = ir = 0.5

        
            
            #--------------------------------------------adicionar aqui as sensiveis
        


            # ## preprocessing and helpers for classification/evaluation
            categorical_fs = [
                "sex",
                "age_cat",
                "race",
                "priors_count",
                "c_charge_degree",
                "score_text",
                "v_score_text",
            ]


            # get the columns in the correct order
            cols = np.concatenate([dataset.columns.copy(deep=True).drop(categorical_fs + [TARGET_NAME]), categorical_fs])
            cols_d = {c: i for i, c in enumerate(cols)}

            classifiers = [
                [RandomForestClassifier, {'random_state': SEED}],
                [DecisionTreeClassifier, {'random_state': SEED}],
                [GaussianNB, {}],
                [LogisticRegression, {}],
                [KNeighborsClassifier, {}],
                [MLPClassifier, {'random_state': SEED}],
            ]
            #-----------------------------------------------------------metodo holdout_ define size de dataset
            # ### Classification
            # setup
            holdout_splits = 1
            holdout = ShuffleSplit(n_splits=holdout_splits, test_size=0.33, random_state=SEED)
            

            #--------------------------------------------------------------aqui define quais as combinaçoes de porporçoes de datasets (na variavel ratios) mas ter cuidado com os pontos dos plots
            rs = [0.01, 0.02, 0.05] + [round(x, 2) for x in np.arange(0.1, 1.0, 0.1)] + [0.95, 0.98, 0.99]
            ratios = [[0.5, ir] for ir in rs] + [[gr, 0.5] for gr in rs]
        
            #--------------------------------------
            # calculations
            fairness_results = []
            results = []
            complexity_values = [["[GR,IR]","F1", "F1v", "F2", "F3", "F4", "R_value", "D3", "CM", "kDN", "T1", "DBC", "N1", "N2", "N3", "N4", "SI", "LSC", "input_noise", "borderline", "deg_overlap", "ICSV", "NSG", "Clust", "ONB"]]
            #complexity_values.append(measures_complexity(X_all,y_all,1,1))                    

            timer.start()

            for gr, ir in ratios:
                print(f'GR: {gr}, IR: {ir}')
                swap_gr, swap_ir = False, False
                c = 0
                for split_seed in seeds_split_data:
                    df = split_data(dataset, SAMPLE_SIZE, gr, ir,split_seed)
                    X_all, y_all = preprocess(df)
                    timer.checkpoint(f"gr={gr} ir={ir} data preprocessing")

                    for i, (traini, testi) in enumerate(holdout.split(X_all)):
                        X_train, X_test = X_all[traini], X_all[testi]
                        y_train, y_test = y_all[traini], y_all[testi]
                        
                        if c == 0:#tirar isto para correr 50 vezes mas demora muito tempo
                            complexity_values.append(measures_complexity(X_test,y_test,gr,ir))
                            c = 1
                            
                        for clf, kwargs in classifiers:
                            pipe = make_pipeline(KNNImputer(), StandardScaler(), clf(**kwargs)).fit(X_train, y_train)
                            f = calculate_fairness(pipe, X_test, y_test, cols_d[SENSIVEL_NAME], group=1 - int(swap_gr), cls=1 - int(swap_ir))

                            for p_metric in [geometric_mean_score, recall_score, f1_score]:
                                results.append(
                                    [
                                        gr,
                                        ir,
                                        clf.__name__.replace('Classifier', ''),
                                        p_metric.__name__,
                                        p_metric(y_test, pipe.predict(X_test), labels=[0, 1]),
                                    ]
                                )

                            for metric, value in f.items():
                                fairness_results.append([gr, ir, clf.__name__.replace('Classifier', ''), metric, value])
                            timer.checkpoint(f"gr={gr} ir={ir} classification with {clf.__name__} rep. {i}")

            results_cv = pd.DataFrame(results, columns=['gr', 'ir', 'clf', 'metric', 'value'])
            fairness_results_cv = pd.DataFrame(fairness_results, columns=['gr', 'ir', 'clf', 'metric', 'value'])
            write_complexity_to_file(complexity_values)
            boxplot(fairness_results_cv,lowerBound_boxplot,upperBound_boxplot)

            timer.checkpoint(f"saving results")
            timer.reset()
            #----------------------------------
            # # pickle the results

            with open(os.path.join(DIRETORIA_EXP, 'fairness_results_cv.pkl'), 'wb') as f:
                pickle.dump(fairness_results_cv, f)

            with open(os.path.join(DIRETORIA_EXP, 'clf_results_cv.pkl'), 'wb') as f:
                pickle.dump(results_cv, f)
            #------------------
            #plot the absolute value of fairness metrics
            for fill in ('std', 'err'):
                subdir = f'line_{fill}'
                os.makedirs(os.path.join(plots_dir, subdir), exist_ok=True)

                for ratio_type, ylim in [
                    ('ir', (-0.9, 0.9)),
                    ('gr', (-0.9, 0.9)),
                ]:
                    for metric in fairness_results_cv['metric'].unique():
                        fig = plot_line(fairness_results_cv, metric, ratio_type, ylim=ylim, fill=fill)
                        fig.savefig(os.path.join(plots_dir, subdir, f'fairness_line_{ratio_type}_{metric}.png'))
                        fig.savefig(os.path.join(plots_dir, subdir, f'fairness_line_{ratio_type}_{metric}.pdf'))
                        plt.close()

            ### plot nan count
            #
            # check how many results are undefined for the metrics and ratios

            #--------------------------------------------
            for ratio_type in ['ir', 'gr']:
                fig = plot_nan(
                    fairness_results_cv,
                    ratio_type,
                    metrics=[
                        'Accuracy Equality Difference',
                        'Statistical Parity Difference',
                        'Equal Opportunity Difference',
                        'Predictive Equality Difference',
                        'Positive Predictive Parity Difference',
                        'Negative Predictive Parity Difference',
                    ],
                )
                fig.savefig(os.path.join(plots_dir, f'fairness_nan_{ratio_type}.png'))
                fig.savefig(os.path.join(plots_dir, f'fairness_nan_{ratio_type}.pdf'))
                plt.close()


            # ## Plot all metrics together

            #------------------------------------------
            timer.start()

            for ratio_type in ['ir', 'gr']:
                fig = plot_line_all(
                    fairness_results_cv,
                    [
                        'Accuracy Equality Difference',
                        'Statistical Parity Difference',
                        'Equal Opportunity Difference',
                        'Predictive Equality Difference',
                        'Positive Predictive Parity Difference',
                        'Negative Predictive Parity Difference',
                    ],
                    ratio_type,
                    fill='std',
                    ylim=(-0.9, 0.9),
                )
                fig.savefig(os.path.join(plots_dir, f'fairness_all_{ratio_type}.png'))
                fig.savefig(os.path.join(plots_dir, f'fairness_all_{ratio_type}.pdf'))
                plt.close()
                timer.checkpoint(f"plotting for {ratio_type}")
            timer.reset()


            # ## Table with classification metrics
            #
            # this code directly prints the tables with formatting for LaTeX

            #-----------------------------------------------
            clfs = results_cv['clf'].unique()
            scores = results_cv['metric'].unique()
            scores_strs = ['ROC AUC', 'G mean', 'recall', 'F1']


            for m, metric in enumerate(scores):
                print(f'\\begin{{tabular}}{{{"l l | " + "c " * len(clfs)}}}')
                print('\\multicolumn{' + str(len(scores) + 2) + '}{c}{' + scores_strs[m] + '} \\\\')
                print('IR & GR & ' + ' & '.join(clfs) + ' \\\\')
                for ratio_type, other_ratio in [['ir', 'gr'], ['gr', 'ir']]:
                    for ratio_val in sorted(results_cv[ratio_type].unique()):
                        subset = results_cv[
                            (results_cv['metric'] == metric)
                            & (results_cv[ratio_type] == ratio_val)
                            & (results_cv[other_ratio] == 0.5)
                        ]
                        if ratio_type == 'ir':
                            print(f'{ratio_val:.2f} & 0.50 ', end='')
                        else:
                            print(f'0.50 & {ratio_val:.2f} ', end='')

                        for clf in clfs:
                            print(
                                f'& {subset[subset["clf"] == clf]["value"].mean():.3f} ({subset[subset["clf"] == clf]["value"].std():.3f}) ',
                                end='',
                            )

                        print('\\\\')
                print('\\end{tabular}\n\n')

            timer.to_file(fn='case_study.csv')


            # # Save to CSV for scatter plots

            #----------------------------------
            # results to csv

            clfs = results_cv['clf'].unique()
            scores = results_cv['metric'].unique()
            scores_strs = ['ROC AUC', 'G mean', 'recall', 'F1']


            for m, metric in enumerate(scores):
                lines = ['IR, GR, ' + ', '.join(clfs)]

                for ratio_type, other_ratio in [['ir', 'gr'], ['gr', 'ir']]:
                    for ratio_val in sorted(results_cv[ratio_type].unique()):
                        subset = results_cv[
                            (results_cv['metric'] == metric)
                            & (results_cv[ratio_type] == ratio_val)
                            & (results_cv[other_ratio] == 0.5)
                        ]
                        if ratio_type == 'ir':
                            l = f'{ratio_val:.2f}, 0.50'
                        else:
                            l = f'0.50, {ratio_val:.2f}'
                        for clf in clfs:
                            l += f'  ,{subset[subset["clf"] == clf]["value"].mean():.3f} ({subset[subset["clf"] == clf]["value"].std():.3f})'
                        lines.append(l)

                with open(os.path.join(DIRETORIA_EXP, f'clf_results_agg_{metric}.csv'), 'w') as f:
                    f.write('\n'.join(lines))

            # fairness results to csv

            clfs = fairness_results_cv['clf'].unique()
            scores = fairness_results_cv['metric'].unique()
            scores_strs = [
                'Accuracy Equality Difference',
                'Statistical Parity Difference',
                'Equal Opportunity Difference',
                'Predictive Equality Difference',
                'Positive Predictive Parity Difference',
                'Negative Predictive Parity Difference',
            ]

            for m, metric in enumerate(scores):
                lines = ['IR, GR; ' + ';'.join(clfs)]

                for ratio_type, other_ratio in [['ir', 'gr'], ['gr', 'ir']]:
                    for ratio_val in sorted(fairness_results_cv[ratio_type].unique()):
                        subset = fairness_results_cv[
                            (fairness_results_cv['metric'] == metric)
                            & (fairness_results_cv[ratio_type] == ratio_val)
                            & (fairness_results_cv[other_ratio] == 0.5)
                        ]
                        if ratio_type == 'ir':
                            l = f'{ratio_val:.2f}, 0.50'
                        else:
                            l = f'0.50, {ratio_val:.2f}'
                        for clf in clfs:
                            l += f'  ;{subset[subset["clf"] == clf]["value"].mean():.3f} ({subset[subset["clf"] == clf]["value"].std():.3f})'
                        lines.append(l)

                with open(os.path.join(DIRETORIA_EXP, f'clf_fairness_agg_{metric}.csv'), 'w') as f:
                    f.write('\n'.join(lines))
