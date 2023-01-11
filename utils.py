import pandas as pd

# Group Ratio
def getGroupRatios(df: pd.DataFrame):
    return (df.j_tp + df.j_fp + df.j_tn + df.j_fn) / (df.i_tp + df.i_fp + df.i_tn + df.i_fn + df.j_tp + df.j_fp + df.j_tn + df.j_fn)

# Imbalance Ratio
def getImbalanceRatios(df: pd.DataFrame):
    return (df.i_tp + df.i_fn + df.j_tp + df.j_fn) / (df.i_tp + df.i_fp + df.i_tn + df.i_fn + df.j_tp + df.j_fp + df.j_tn + df.j_fn)

# Stereotipical bias
def getStereotipicalBias(df: pd.DataFrame):
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
def getPositivePredictiveValue_i(df: pd.DataFrame):
    return df.i_tp / (df.i_tp + df.i_fp)
def getPositivePredictiveValue_j(df: pd.DataFrame):
    return df.j_tp / (df.j_tp + df.j_fp)

# Negative Predictive Value
def getNegativePredictiveValue_i(df: pd.DataFrame):
    return df.i_tn / (df.i_tn + df.i_fn)
def getNegativePredictiveValue_j(df: pd.DataFrame):
    return df.j_tn / (df.j_tn + df.j_fn)

# Statistical Parity
# each group has the same probability of being classified with a positive outcome
def get_statistical_parity(df: pd.DataFrame):
    return ((df.j_tp + df.j_fp)/(df.j_tp + df.j_fp + df.j_tn + df.j_fn)) - ((df.i_tp + df.i_fp)/(df.i_tp + df.i_fp + df.i_tn + df.i_fn))

# Disparate Impact
# similiar to statistical parity, but using ratio
def get_disparate_impact(df: pd.DataFrame):
    return ((df.j_tp + df.j_fp)/(df.j_tp + df.j_fp + df.j_tn + df.j_fn)) / ((df.i_tp + df.i_fp)/(df.i_tp + df.i_fp + df.i_tn + df.i_fn))

# Accuracy Equality Ratio
def get_acc_equality_ratio(df: pd.DataFrame):
    return ((df.j_tp + df.j_tn)/(df.j_tp + df.j_fp + df.j_tn + df.j_fn)) / ((df.i_tp + df.i_tn)/(df.i_tp + df.i_fp + df.i_tn + df.i_fn))

# Accuracy Equality Difference
def get_acc_equality_diff(df: pd.DataFrame):
    return ((df.j_tp + df.j_tn)/(df.j_tp + df.j_fp + df.j_tn + df.j_fn)) - ((df.i_tp + df.i_tn)/(df.i_tp + df.i_fp + df.i_tn + df.i_fn))

# Equal Opportunity Ratio
def get_equal_opp_ratio(j_tpr, i_tpr):
    return j_tpr / i_tpr

# Equal Opportunity Difference
def get_equal_opp_diff(j_tpr, i_tpr):
    return j_tpr - i_tpr

# Predictive Equality Ratio
def get_pred_equality_ratio(j_fpr, i_fpr):
    return j_fpr / i_fpr

#Predictive Equality Difference
def get_pred_equality_diff(j_fpr, i_fpr):
    return j_fpr - i_fpr

# Positive Predictive Parity Ratio
def get_pred_parity_ratio(j_ppv, i_ppv):
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