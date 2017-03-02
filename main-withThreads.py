import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

from utils import nnz_fraction
from base import OnlineClassifier
from RDA import *
from FTRLProx import FollowTheRegularizedLeaderProximal
from Fobos import FOBOS

from threading import Thread

def loading_dataset(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]

def main(datafile):
    X, y = loading_dataset(datafile)

    # Change -1 values to 0
    #y[y == -1] = 0

    # Remove zeros entries
    nnz_entries = np.unique(X.nonzero()[0])
    X = X[nnz_entries]
    y = y[nnz_entries]

    # TODO: define an OnlineClassifier instance and train it over the dataset

    return X, y

def execThread(lbda1):
        start_time = datetime.now()
        print (" ##### lbda1 = %f" %lbda1)
        #FTRL = FollowTheRegularizedLeaderProximal(lbda1=lbda1)
        #w, y_proba = FTRL.train(X_sub,y_sub)
        fobos = FOBOS(initialization='random', loss='logloss',
              lamda1= lbda1, regularization='l1', initial_step=.75, with_log=True)
        for i in range(np.shape(X_sub)[0]):
            x_t = X_sub[i].toarray()[0,:]
            y_t = y_sub[i]
            #print(x_t)
            if i%100 == 0:
                print("progress :%d/10 \t [worker: lambda=%s ]"%(i/100, lbda1))
            fobos.fit(x_t, y_t)
        y_proba = fobos.probas
        w = fobos.w

        roc = roc_auc_score(y_subb, y_proba)
        nnz = nnz_fraction(w)
        print('ROC: %f | ' 'NNZ: %f | ' 'Time taken: %s seconds'
           % (roc, nnz, (datetime.now() - start_time).seconds))
        roc_score.append(roc)
        nnz_frac.append(nnz)

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
    y_subb = y_sub.copy()
    y_subb[y_subb == -1] = 0

    roc_score = []
    nnz_frac = []
    # FTRL Prox
    lbda1s = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

    threads = []
    for lbda1 in lbda1s:
        th = Thread(target=execThread, args=[lbda1])
        threads.append(th)
        th.start()

    print("[Waiting for threads to finish]")

    for th in threads:
        th.join()



    plot_df = pd.DataFrame({'ROC': roc_score, 'NNZ':nnz_frac})
    plot_df.to_csv('results/FOBOS-result-news20.csv', index=None)

    plt.plot(roc_score,nnz_frac)
    plt.gca().invert_yaxis()
    plt.savefig('plots/Fobos-plot-news20.png')
    plt.show()


