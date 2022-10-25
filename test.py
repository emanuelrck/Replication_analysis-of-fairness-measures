import glob
import re
import numpy as np
import pickle

def binData2array(fname):
    fname
    with open(fname, 'rb') as f:
        X = pickle.load(f)
    return X

# Test if each row sums to `n`
for f in glob.glob('*'):
    if (re.match(r'n.*\.bin$', f)):
        n = int(f.strip('n').strip('.bin'))
        x = binData2array(f)
        for row in x:
            assert np.sum(row) == n
