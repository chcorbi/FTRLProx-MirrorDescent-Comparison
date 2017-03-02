import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score

from utils import nnz_fraction
from RDA import RDASolver
from FTRLProx import FollowTheRegularizedLeaderProximal


def loading_dataset(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]


def main(datafile):
    X, y = loading_dataset(datafile)

    # Change -1 values to 0
    y[y == -1] = 0

    # Remove zeros entries
    nnz_entries = np.unique(X.nonzero()[0])
    X = X[nnz_entries]
    y = y[nnz_entries]
    return X, y


def plotandsave(algo, filename, roc, nnz):
    plot_df = pd.DataFrame({'ROC': roc, 'NNZ': nnz})
    plot_df.to_csv('results/' + algo + '-result-' + filename, index=None)
    plt.figure()
    plt.plot(roc, nnz)
    plt.gca().invert_yaxis()
    plt.savefig('plots/' + algo +'-plot-' + filename + '.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help="One input file")
    args = parser.parse_args()

    # Get X, y
    X, y = main(args.input_file)
    filename = args.input_file.split('/')[-1]

    # Subsample
    N = 1000
    np.random.seed(42)
    i = np.random.choice(np.arange(X.shape[0]), N, replace=False)
    X_sub = X[i]
    y_sub = y[i]

    roc_ftrlp = []
    nnz_ftrlp = []
    roc_rda = []
    nnz_rda = []
    roc_fobos = []
    nnz_fobos = []

    lbda1s = [1e-5, 1e-2, 1e-1, 1]

    for lbda1 in lbda1s:
        start_time = datetime.now()
        print(" ##### lbda1 = %f" % lbda1)

        # FTRL
        FTRL = FollowTheRegularizedLeaderProximal(lbda1=lbda1)
        w, y_proba = FTRL.train(X_sub, y_sub)
        roc = roc_auc_score(y_sub, y_proba)
        nnz = nnz_fraction(w)
        print('ROC: %f | ' 'NNZ: %f | ' 'Time taken: %s seconds'
              % (roc, nnz, (datetime.now() - start_time).seconds))
        roc_ftrlp.append(roc)
        nnz_ftrlp.append(nnz)

        # RDA
        RDA = RDASolver(lbda1=lbda1, gamma=2.0)
        w, y_proba = RDA.train(X_sub, y_sub)
        roc = roc_auc_score(y_sub, y_proba)
        nnz = nnz_fraction(w)
        print('ROC: %f | ' 'NNZ: %f | ' 'Time taken: %s seconds'
              % (roc, nnz, (datetime.now() - start_time).seconds))
        roc_rda.append(roc)
        nnz_rda.append(nnz)

        # FOBOS
        FOBOS = FOBOS(lbda1=lbda1, gamma=2.0)
        w, y_proba = FOBOS.train(X_sub, y_sub)
        roc = roc_auc_score(y_sub, y_proba)
        nnz = nnz_fraction(w)
        print('ROC: %f | ' 'NNZ: %f | ' 'Time taken: %s seconds'
              % (roc, nnz, (datetime.now() - start_time).seconds))
        roc_fobos.append(roc)
        nnz_fobos.append(nnz)

    plotandsave('FTRLP', filename, roc_ftrlp, nnz_ftlrp)
    plotandsave('RDA', filename, roc_rda, nnz_rda)
    plotandsave('FOBOS', filename, roc_fobos, nnz_fobos)

