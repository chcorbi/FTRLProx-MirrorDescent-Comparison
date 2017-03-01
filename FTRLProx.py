import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator
from utils import log_loss, sigmoid


class FollowTheRegularizedLeaderProximal (BaseEstimator):
    '''
    Follow The Regularied Leader Proximal
    minimizes iteratively with an adaptive combination of L2 and L1 norms.
    '''
    
    def __init__(self, alpha=1., beta=1., lbda1=1., lbda2=1., verbose=1):
        # Learning rate's proportionality constant.
        self.alpha = alpha
        # Learning rate parameter.
        self.beta = beta
        # L1 regularization parameter.
        self.lbda1 = lbda1
        # L2 regularization parameter.
        self.lbda2 = lbda2
        
        #Initialize weights parameters
        self.z = None
        self.n = None
        
        # Loss initialization
        self.log_likelihood = 0
        self.loss = []
        
        self.verbose=verbose


    def train(self, X, y):
        start_time = datetime.now()
                       
        self.z = [0.] * X.shape[1]
        self.n = [0.] * X.shape[1]

        y_proba = []

        for t in range(X.shape[0]):
            # Init weight vector
            w = {}

            # Init dot product
            wtx = 0

            # Non-zeros features of X[t]
            I = X[t].nonzero()[1]

            # Security
            if  I.size == 0:
                raise "Error at ligne %d " %(t+1)
                continue

            # Update weight
            for i in I:
                if self.z[i] <= self.lbda1:
                    w[i] = 0
                else:
                    sign = 1. if self.z[i] >= 0 else -1.
                    w[i] = - (self.z[i] - sign * self.lbda1) / ((self.beta + np.sqrt(self.n[i])) / self.alpha + self.lbda2)

                # Compute dot product
                wtx += w[i] * X[t,i] 

            # Predict output probability
            p = sigmoid(wtx)

            # Update weights parameters
            for i in I:
                # Compute gradient of loss w.r.t wi
                g_i = (p - y[t]) * X[t,i]

                # Update sigma_i
                sigma_i = (np.sqrt(self.n[i] + g_i * g_i) - np.sqrt(self.n[i])) / self.alpha

                # Update weights parameters
                self.z[i] += g_i - sigma_i * w[i]
                self.n[i] += g_i * g_i

            # Compute loss
            self.log_likelihood += log_loss(y[t], p)

            # Append to the loss list.
            self.loss.append(self.log_likelihood)

            # Print all the current information
            if (self.verbose==1 and t%(X.shape[0]/10)==0):
                print('Training Samples: {0:9} | ' 'Loss: {1:11.2f}'
                  .format(t, self.log_likelihood, (datetime.now() - start_time).seconds))

            # Add proba
            y_proba.append(p)
            
        return w, y_proba
