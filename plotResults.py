import matplotlib.pyplot as plt
import pandas as pd

ftrlp = pd.read_csv('results/FTRLP-result-news20.csv')
roc_ftrlp = ftrlp['ROC'].values
nnz_ftlrp = ftrlp['NNZ'].values

fobos = pd.read_csv('results/FOBOS-result-rcv1.csv')
roc_fobos = fobos['ROC'].values
nnz_fobos = fobos['NNZ'].values

rda = pd.read_csv('results/RDA-result-rcv1_train.csv')
roc_rda = rda['ROC'].values
nnz_rda = rda['NNZ'].values

plt.figure()
plt.plot(roc_ftrlp, nnz_ftlrp, label='FTRLP')
plt.plot(roc_fobos, nnz_fobos, label='FOBOS')
plt.plot(roc_rda, nnz_rda, label='RDA')
plt.gca().invert_yaxis()

plt.xlabel("ROC")
plt.ylabel("NNZ fraction")
plt.legend(loc='best')

plt.show()