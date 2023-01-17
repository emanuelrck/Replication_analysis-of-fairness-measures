# Analysis Of Fairness Measures

![Test](https://github.com/alessandro1802/analysis-of-fairness-measures/actions/workflows/test.yml/badge.svg)
[![Python](https://shields.io/badge/python-v3.11-blue)](https://www.python.org/downloads/)

## Reference

The project's synthetic dataset was generated using the code from [this repository](https://github.com/Rineol/fairness-measures). Hence, the generator is not present here.

## The goal

1. be able to choose the right measure of fairness depending on various conditions

2. eventually, answer whether the data is balanced (or how much it is probable to be fair)

## Set-up

- After cloning the repository, create a *virtual environment* inside: `python3.11 -m venv venv`

- Don't forget to activate it (depending on your OS) e.g. `source venv/bin/activate`

- Install the dependencies: `pip install -r requirements.txt`

## File-structure

**Remark**: related to `n = 24` dataset will be referenced to as *sample*, while `n = 56`, which we worked with, will be *main*.

- `utils` script contains (mainly metric) functions

- `metricCalculation` is how we got the calculations

- `calculations` directory contains *sample* metric calculations

- `plotting` is code for getting plots out of calculations

- `plots` directory contains 2 sub-directories:
  
  1. `n24` are *sample* plots
  
  2. `n56` are *main* plots

- `data` directory contains the *sample* dataset itself

- `resourseConsumption` is experiments on time and memory complexity for generating datasets
