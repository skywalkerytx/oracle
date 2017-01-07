import xgboost as xgb
import numpy as np
import pyut
from sklearn.model_selection import train_test_split


def param():
    clfparam = dict()
    clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 0
    clfparam['max_depth'] = 6
    clfparam['subsample'] = 0.5
    return clfparam


if __name__ == '__main__':
    # vectors, labels = pyut.data(labelnum=0)

    # tvector, evector, tlabel, elabel = train_test_split(vectors, labels, test_size=0.2, random_state=42)
    tm = xgb.DMatrix('data/xgb/trainingdata.matrix')
    em = xgb.DMatrix('data/xgb/testingdata.matrix')
    # tm = xgb.DMatrix(tvector, label=tlabel)
    # em = xgb.DMatrix(evector, label=elabel)
    # tm.save_binary('data/xgb/trainingdata.matrix')
    # em.save_binary('data/xgb/testingdata.matrix')

    tl = tm.get_label()
    el = em.get_label()

    print(len(list(filter(lambda x: x == 1, tl))) + len(list(filter(lambda x: x == 1, el))))

    print(len(list(filter(lambda x: x == 0, tl))) + len(list(filter(lambda x: x == 0, el))))

    print(len(tl)+len(el))

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