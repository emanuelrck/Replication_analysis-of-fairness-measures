import glob
import numpy as np
import pickle

def loadDataSet(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

# Test if each row sums to `n`
for f in glob.glob("../data/*.bin"):
    n = int(f.split('/')[-1].strip('n').strip('.bin'))
    dataset = loadDataSet(f)
    for row in dataset:
        assert np.sum(row) == n
