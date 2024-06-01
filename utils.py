__all__ = [
    'get_group_ratios',
    'get_imbalance_ratios',
    'get_stereotypical_bias',
    'getTPR_i',
    'getTPR_j',
    'getFPR_i',
    'getFPR_j',
    'get_positive_predictive_value_i',
    'get_positive_predictive_value_j',
    'get_negative_predictive_value_i',
    'get_negative_predictive_value_j',
    'get_statistical_parity',
    'get_disparate_impact',
    'get_acc_equality_ratio',
    'get_acc_equality_diff',
    'get_equal_opp_ratio',
    'get_equal_opp_diff',
    'get_pred_equality_ratio',
    'get_pred_equality_diff',
    'get_pos_pred_parity_ratio',
    'get_pos_pred_parity_diff',
    'get_neg_pred_parity_ratio',
    'get_neg_pred_parity_diff',
    'Timer',
]

import pandas as pd
from os import path
from time import perf_counter


# Group Ratio
def get_group_ratios(df: pd.DataFrame):
    return (df.j_tp + df.j_fp + df.j_tn + df.j_fn) / (
                df.i_tp + df.i_fp + df.i_tn + df.i_fn + df.j_tp + df.j_fp + df.j_tn + df.j_fn)


# Imbalance Ratio
def get_imbalance_ratios(df: pd.DataFrame):
    return (df.i_tp + df.i_fn + df.j_tp + df.j_fn) / (
                df.i_tp + df.i_fp + df.i_tn + df.i_fn + df.j_tp + df.j_fp + df.j_tn + df.j_fn)


# Stereotypical bias
def get_stereotypical_bias(df: pd.DataFrame):
    return (df.i_tp + df.i_fp + df.i_tn + df.i_fn) / (df.j_tp + df.j_fp + df.j_tn + df.j_fn) - \
        (df.j_tp + df.j_fp + df.j_tn + df.j_fn) / (df.i_tp + df.i_fp + df.i_tn + df.i_fn)


# True Positive Rate
def getTPR_i(df: pd.DataFrame):
    return df.i_tp / (df.i_tp + df.i_fn)


def getTPR_j(df: pd.DataFrame):
    return df.j_tp / (df.j_tp + df.j_fn)


# False Positive Rate
def getFPR_i(df: pd.DataFrame):
    return df.i_fp / (df.i_fp + df.i_tn)


def getFPR_j(df: pd.DataFrame):
    return df.j_fp / (df.j_fp + df.j_tn)


# Positive Predictive Value
def get_positive_predictive_value_i(df: pd.DataFrame):
    return df.i_tp / (df.i_tp + df.i_fp)


def get_positive_predictive_value_j(df: pd.DataFrame):
    return df.j_tp / (df.j_tp + df.j_fp)


# Negative Predictive Value
def get_negative_predictive_value_i(df: pd.DataFrame):
    return df.i_tn / (df.i_tn + df.i_fn)


def get_negative_predictive_value_j(df: pd.DataFrame):
    return df.j_tn / (df.j_tn + df.j_fn)


# Statistical Parity
# each group has the same probability of being classified with a positive outcome
def get_statistical_parity(df: pd.DataFrame):
    return ((df.j_tp + df.j_fp) / (df.j_tp + df.j_fp + df.j_tn + df.j_fn)) - (
                (df.i_tp + df.i_fp) / (df.i_tp + df.i_fp + df.i_tn + df.i_fn))


# Disparate Impact
# similiar to statistical parity, but using ratio
def get_disparate_impact(df: pd.DataFrame):
    return ((df.j_tp + df.j_fp) / (df.j_tp + df.j_fp + df.j_tn + df.j_fn)) / (
                (df.i_tp + df.i_fp) / (df.i_tp + df.i_fp + df.i_tn + df.i_fn))


# Accuracy Equality Ratio
def get_acc_equality_ratio(df: pd.DataFrame):
    return ((df.j_tp + df.j_tn) / (df.j_tp + df.j_fp + df.j_tn + df.j_fn)) / (
                (df.i_tp + df.i_tn) / (df.i_tp + df.i_fp + df.i_tn + df.i_fn))


# Accuracy Equality Difference
def get_acc_equality_diff(df: pd.DataFrame):
    return ((df.j_tp + df.j_tn) / (df.j_tp + df.j_fp + df.j_tn + df.j_fn)) - (
                (df.i_tp + df.i_tn) / (df.i_tp + df.i_fp + df.i_tn + df.i_fn))


# Equal Opportunity Ratio
def get_equal_opp_ratio(j_tpr, i_tpr):
    return j_tpr / i_tpr


# Equal Opportunity Difference
def get_equal_opp_diff(j_tpr, i_tpr):
    return j_tpr - i_tpr


# Predictive Equality Ratio
def get_pred_equality_ratio(j_fpr, i_fpr):
    return j_fpr / i_fpr


# Predictive Equality Difference
def get_pred_equality_diff(j_fpr, i_fpr):
    return j_fpr - i_fpr


# Positive Predictive Parity Ratio
# renamed from get_pred_parity_ratio
def get_pos_pred_parity_ratio(j_ppv, i_ppv):
    return j_ppv / i_ppv


# Positive Predictive Parity Difference
def get_pos_pred_parity_diff(j_ppv, i_ppv):
    return j_ppv - i_ppv


# Negative Predictive Parity Ratio
def get_neg_pred_parity_ratio(j_npv, i_npv):
    return j_npv / i_npv


# Negative Predictive Parity Difference
def get_neg_pred_parity_diff(j_npv, i_npv):
    return j_npv - i_npv


class Timer:
    def __init__(self):
        self.records = list()
        self.current = None
        self.start_t = None
        self.last_t = None

    def start(self):
        if self.current:
            self.records.append(self.current)
        self.current = dict()
        self.start_t = perf_counter()
        self.last_t = perf_counter()

        return self

    def checkpoint(self, label: str, from_start=False):
        assert self.last_t is not None, "Start the timer first!"

        t = perf_counter()
        if from_start:
            self.current[label] = t - self.start_t
        else:
            self.current[label] = t - self.last_t
        self.last_t = t

        return self.current[label]

    def reset(self):
        if self.current:
            self.records.append(self.current)
        self.current = None
        self.start_t = None
        self.last_t = None

    def print(self):
        for rec in self.records:
            for k, v in rec.items():
                print(f'{k}:\t {v}')
            print('\n')

    def to_file(self, fn: str):
        timer_dir = "out/time/"
        with open(timer_dir + fn, 'a') as f:
            for rec in self.records:
                for k, v in rec.items():
                    f.write(f'{k}, {v}\n')
                f.write('\n')
            f.write('\n\n')
