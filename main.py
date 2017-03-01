import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score

from utils import nnz_fraction
from solvers import RDASolver
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

    # TODO: define an OnlineClassifier instance and train it over the dataset

    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help="One input file")
    args = parser.parse_args()

    # Get X, y
    X, y = main(args.input_file)

    # Subsample
    N = 1000
    np.random.seed(42)
    i = np.random.choice(np.arange(X.shape[0]), N, replace=False)
    X_sub = X[i]
    y_sub = y[i]

    roc_score = []
    nnz_frac = []
    # FTRL Prox
    lbda1s = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

    for lbda1 in lbda1s:
        start_time = datetime.now()
        print(" ##### lbda1 = %f" % lbda1)
        FTRL = FollowTheRegularizedLeaderProximal(lbda1=lbda1)
        w, y_proba = FTRL.train(X_sub, y_sub)
        roc = roc_auc_score(y_sub, y_proba)
        nnz = nnz_fraction(w)
        print('ROC: %f | ' 'NNZ: %f | ' 'Time taken: %s seconds'
              % (roc, nnz, (datetime.now() - start_time).seconds))
        roc_score.append(roc)
        nnz_frac.append(nnz)

    plot_df_ftrl = pd.DataFrame({'ROC': roc_score, 'NNZ': nnz_frac})
    plot_df_ftrl.to_csv('results/FTRLP-result-news20.csv', index=None)

    plt.figure()
    plt.plot(roc_score, nnz_frac)
    plt.gca().invert_yaxis()
    plt.savefig('plots/FTRLP-plot-news20.png')

    # RDA
    print("")
    print("")
    print("RDA")
    roc_score = []
    nnz_frac = []

    for lbda1 in lbda1s:
        start_time = datetime.now()
        print(" ##### lbda1 = %f" % lbda1)
        rda = RDASolver(lbda=lbda1, gamma=1.0)
        w, y_proba = rda.train(X_sub, y_sub)
        roc = roc_auc_score(y_sub, y_proba)
        nnz = w.nnz / w.shape[1]
        print('ROC: %f | ' 'NNZ: %f | ' 'Time taken: %s seconds'
              % (roc, nnz, (datetime.now() - start_time).seconds))
        roc_score.append(roc)
        nnz_frac.append(nnz)

    plot_df_rda = pd.DataFrame({'ROC': roc_score, 'NNZ': nnz_frac})
    plot_df_rda.to_csv('results/RDA-result-news20.csv', index=None)

    plt.figure()
    plt.plot(roc_score, nnz_frac)
    plt.gca().invert_yaxis()
    plt.savefig('plots/RDA-plot-news20.png')
    plt.show()
