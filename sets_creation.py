import os
import time
import sys
import math
import numpy as np
import pickle
from copy import deepcopy
from enum import StrEnum, auto


class Action(StrEnum):
    GENERATE = auto()
    LOAD = auto()
    SAVE_BIN = auto()
    SAVE_TXT = auto()
    HELP = auto()


def parse_args(argv):
    args = deepcopy(argv[1:])
    config = dict()

    while len(args):
        match a := args.pop(0).lower():
            case '-g' | '--generate':
                config[Action.GENERATE] = (args.pop(0), args.pop(0))
            case  '-l' | '--load':
                config[Action.LOAD] = args.pop(0)
            case '-b' | '--save-binary':
                config[Action.SAVE_BIN] = args.pop(0)
            case '-t' | '--save-txt':
                config[Action.SAVE_TXT] = args.pop(0)
            case '-h' | '--help':
                config[Action.HELP] = True
                print(
                    'Available options:',
                    '  -g <n> <k> | --generate: generate a dataset of given size',
                    '  -l <file> | --load <file>: load data from file',
                    '  -b <file> | --save-binary <file>: save to the file (binary)',
                    '  -t <file> | --save-txt <file>: save to the file (human-readable)',
                    'e.g.:',
                    '  python3 sets_creation.py -g 8 56 -b "Set(08,56).bin"',
                    '',
                    sep='\n'
                )
            case _:
                print(f'Invalid argument: {a}')
    return config


def the_ratio(n, k):
    m = 1
    for i in range(1, n):
        m = m * (k + i)
    m = round(m / math.factorial(n - 1))
    return m


def genset_k_do_increment(k, X):
    a = np.shape(X)
    lastNZ = np.amax(np.matmul((X > 0), np.diag(range(1, k + 1))), axis=1)
    b = np.shape(lastNZ)
    k1lastNZ = k + 1 - lastNZ
    Y = np.repeat(X, k1lastNZ, axis=0)
    c = np.shape(Y)
    idxY = 1
    for q in range(1, a[0] + 1):
        cs = lastNZ[q - 1]
        for c in range(cs, k + 1):
            Y[idxY + c - cs - 1, c - 1] += 1
        idxY = idxY + k + 1 - cs
    return Y


def genset_k_by_inc(n, k):
    m = the_ratio(n, k)

    X = np.zeros((m, n), dtype=np.int8, order='C')
    for i in range(0, n):
        X[i][i] = 1
    sm = 1
    mX = the_ratio(n, sm)
    while sm < k:
        tm = time.time()
        sm += 1
        mX1 = the_ratio(n, sm)
        X[0:mX1 - 1, :] = genset_k_do_increment(n, X[0:mX - 1, :])
        mX = mX1
        print(f'iteration: {k - sm} -- {time.time() - tm:.2f} [s]')

    X[-1, -1] = X[0, 0]
    return X


def generate_dataset(n, k):
    start_time = time.time()
    print('Generating simplex data', end='')
    print('\n')
    X = genset_k_by_inc(n, k)
    print(' -- Done')

    mn = np.shape(X)
    print(np.sum(X[0, :]))
    print(mn)
    print(f'GenSimpBIN: {time.time() - start_time:.2f} [s]')
    return X


def save_bin_dataset(X, fname):
    start_time = time.time()
    print(f'Saving BIN file: {fname}', end='')
    with open(fname, 'wb') as f:
        pickle.dump(X, f)
    print(f'SaveBIN: {time.time() - start_time:.2f} [s]')
    print(' -- Done')
    return


def save_txt_dataset(X, fname):
    start_time = time.time()
    print(f'Saving TXT file: {fname}', end='')
    np.savetxt(fname, X, fmt='%i')
    print(' -- Done')
    print(f'SaveTXT: {time.time() - start_time:.2f} [s]')
    return


def load_bin_dataset(fname):
    print(f'Loading BIN file: {fname}', end='')
    with open(fname, 'rb') as f:
        X = pickle.load(f)
    print(' -- Done')

    mn = np.shape(X)
    print(np.sum(X[0, :]))
    print(mn)

    return X


def load_txt_dataset(fname):
    print(f'Loading TXT file: {fname}', end='')
    X = np.loadtxt(fname, dtype=np.int8)
    print(' -- Done')

    mn = np.shape(X)
    print(np.sum(X[0, :]))
    print(mn)

    return X


if __name__ == '__main__':
    conf = parse_args(sys.argv)

    if Action.HELP in conf:
        exit(0)

    assert Action.GENERATE in conf.keys() or Action.LOAD in conf.keys(), \
        'The data must be either generated or loaded from the file.'

    start_exec_time = time.time()

    if Action.GENERATE in conf.keys():
        print('Starting: generate dataset...')
        start_task_time = time.time()
        n, k = conf[Action.GENERATE]
        X = generate_dataset(n, k)
        print(f'Dataset generated in {time.time() - start_task_time}s')

    elif f := conf.get(Action.LOAD):
        print('Starting: load dataset from file...')
        start_task_time = time.time()
        X = load_bin_dataset(os.path.join('out', f))
        print(f'Dataset loaded in {time.time() - start_task_time}s')

    if f := conf.get(Action.SAVE_BIN):
        print('Starting: save dataset to binary file...')
        start_task_time = time.time()
        os.makedirs('out', exist_ok=True)
        save_bin_dataset(X, os.path.join('out', f))
        print(f'Dataset saved in {time.time() - start_task_time}s')

    if f := conf.get(Action.SAVE_TXT):
        print('Starting: save dataset to text file...')
        start_task_time = time.time()
        os.makedirs('out', exist_ok=True)
        save_txt_dataset(X, os.path.join('out', f))
        print(f'Dataset saved in {time.time() - start_task_time}s')

    print(f'Total duration: {time.time() - start_exec_time}s')
