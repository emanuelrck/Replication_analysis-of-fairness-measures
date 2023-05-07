# ECAI paper #320

Properties of fairness measures in the context of varying class imbalance and protected group ratios

# Reproducibility

## Environment

The experiments were conducted on a machine equipped with Intel® Core™ i7-1260P 4.7GHz processor
and 48 GB of RAM, using Python 3.10.10, on Ubuntu 22.04.

The required packages, with specified versions, are listed in `requirements.txt`
and can be installed using `pip install -r requirements.txt`.

## Repository contents

### Experiments with synthetic data (Secitons 3 and 4 of the paper)

- `sets_creation.py`: generation of synthetic data, consisting of all possible confusion matrices with regard to
    the protected groups and decision classes.
- `metrics_calculations`: calculation of fairness measures for synthetic data
- `histograms_plot.py`: distribution of fairness measures
- `perfect_fairness_and_undefined.py`: probability of perfect fairness and undefined values of metrics

### Experiments with real-world data (Section 5)

- `case_study.py`: all the code for the case study with the Adult dataset

## Code execution

For the experiments with synthetic data, `sets_creation.py` and `metrics_calculations` need to be run first
(in this specific order). The remaining scripts can be run in any order.

For the experiments with real-world data, `case_study.py` requires the file `adult.data` from the
[UCI repository](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/) to be present under `data/`.
