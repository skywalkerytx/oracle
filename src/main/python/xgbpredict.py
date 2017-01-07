import xgboost as xgb
import numpy as np
import pyut
from sklearn.model_selection import train_test_split


def param():
    clfparam = dict()
    clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 0
    clfparam['max_depth'] = 8
    clfparam['subsample'] = 0.5
    return clfparam


if __name__ == '__main__':
    tm,em = pyut.sourcefromfile()

    watchlist = [(em, 'eval'), (tm, 'train')]
    num_round = 2
    bst = xgb.train(param(), tm, num_round, watchlist)
    preds = bst.predict(em)
    elabel = em.get_label()
    positive = 0
    detected = 0
    falsealarm = 0
    print('total ' + str(len(preds)))
    for i in range(0, len(preds)):
        if preds[i] > 0.5:
            detected += 1
            if elabel[i] == 1:
                positive += 1
            else:
                falsealarm += 1
    print(positive * 100.0 / detected, falsealarm * 100.0 / detected)