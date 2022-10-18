import time
import sys
import math
import numpy as np
import pickle


def the_ratio(n, k):
    m = 1
    for i in range(1, n):
        m = m * (k + i)
    m = round(m / math.factorial(n - 1))
    return m


def genset_k_do_increment(k, X):
    a = np.shape(X)
    lastNZ = np.amax(np.matmul((X>0), np.diag(range(1, k+1))), axis=1)
    b = np.shape(lastNZ)
    k1lastNZ = k+1-lastNZ
    Y = np.repeat(X,k1lastNZ, axis=0)
    c = np.shape(Y)
    idxY = 1
    for q in range(1,a[0]+1):
        cs = lastNZ[q-1]
        for c in range(cs,k+1):
            Y[idxY+c-cs-1,c-1] += 1
        idxY = idxY + k+1-cs
    return Y


def genset_k_by_inc(n, k):
    m = the_ratio(n, k)

    X = np.zeros((m, n), dtype=np.int8, order='C')
    for i in range(0, n):
        X[i][i] = 1
    sm = 1 
    mX = the_ratio(n, sm)
    while (sm < k):
        tm = time.time()
        sm = sm + 1
        mX1 = the_ratio(n, sm)
        X[0:mX1-1, :] = genset_k_do_increment(n, X[0:mX - 1, :])
        mX = mX1
        print("iteration:%i -- %.2f [s]" % (k - sm, time.time() - tm))
        
    X[-1,-1] = X[0,0]
    return X


def generate_dataset(n, k):
    start_time = time.time()
    print("Generating simplex data",end='')
    print('\n')
    X = genset_k_by_inc(n, k)
    print(" -- Done")

    mn = np.shape(X)
    print(np.sum(X[0, :]))
    print(mn)
    print("GenSimpBIN: %.2f [s]" % (time.time() - start_time))
    return X


def save_bin_dataset(X, fname):
    start_time = time.time()
    print("Saving BIN file: %s" % fname, end='')
    with open(fname, 'wb') as f:
        pickle.dump(X, f)
    print("SaveBIN: %.2f [s]" % (time.time() - start_time))
    print(" -- Done")
    return


def save_txt_dataset(X, fname):
    start_time = time.time()
    print("Saving TXT file: %s" % fname, end='')
    np.savetxt(fname, X, fmt='%i')
    print(" -- Done")
    print("SaveTXT: %.2f [s]" % (time.time() - start_time))
    return


def load_bin_dataset(fname):
    print("Loading BIN file: %s" % fname, end='')
    with open(fname, 'rb') as f:
        X = pickle.load(f)
    print(" -- Done")

    mn = np.shape(X)
    print(np.sum(X[0, :]))
    print(mn)

    return X


def load_txt_dataset(fname):
    print("Loading TXT file: %s" % fname, end='')
    X = np.loadtxt(fname, dtype=np.int8)
    print(" -- Done")

    mn = np.shape(X)
    print(np.sum(X[0, :]))
    print(mn)

    return X


if __name__ == '__main__':

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        k = int(sys.argv[1])
        print('Comand line params: ', end='')
    else:
        # k - size of set; n -  error matrix combinations
        # k should be a multiple of n, (otherwise a potentially incomplete data set is generated)
        n = 8
        k = n * 5
        print('Default params: ', end='')
    
    print('n=%i, k=%i'%(n,k))
    prog_start_time = time.time()
    bin_fname = "Set(%02i,%02i).bin" % (n, k)

    # Data generating
    # X = generate_dataset(n, k)
    
    # Data saving - bin
    # save_bin_dataset(X, bin_fname)

    # Data loading - bin
    # X = load_bin_dataset(bin_fname)
    
    # Data saving - txt
    # txt_fname = "Set(%02i,%02i).txt" % (n, k)
    # save_txt_dataset(X, txt_fname)
            
    print("Total time: %.2f [s]" % (time.time() - prog_start_time))
