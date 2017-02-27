import argparse

import numpy as np
from scipy import stats
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from utils import nnz_fraction
from base import OnlineClassifier
from solvers import *
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

    print('y:', stats.describe(y))
    # TODO: define an OnlineClassifier instance and train it over the dataset

    return X, y




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help="One input file")
    args = parser.parse_args()
    
    X, y = main(args.input_file)

    # FTRL Prox 
    FTRL = FollowTheRegularizedLeaderProximal()
    w, y_proba = FTRL.train(X[:1000,:],y[:1000])
    print ("ROC Score: %f" %roc_auc_score(y[:1000], y_proba))
    print ("Non zero fraction of final weight: %f" %nnz_fraction(w))
