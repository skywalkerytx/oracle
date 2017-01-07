import xgboost as xgb
import numpy as np
import pyut
from sklearn.model_selection import train_test_split

def param():
    clfparam = {}
    clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 0
    clfparam['max_depth'] = 6
    clfparam['subsample'] = 0.5
    return clfparam

if __name__ == '__main__':
    vectors, labels = pyut.data(labelnum=0)

    tvector , evector, tlabel, elabel = train_test_split(vectors, labels, test_size=0.2,random_state=42)
    tm = xgb.DMatrix(tvector, label=tlabel)
    em = xgb.DMatrix(evector, label=elabel)
    watchlist = [(em,'eval'),(tm,'train')]
    num_round = 2
    bst = xgb.train(param(),tm,num_round,watchlist)

    preds = bst.predict(em)
    labels = em.get_label()
    print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
    print(tlabel)
