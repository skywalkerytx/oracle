import xgboost as xgb
from multiprocessing.pool import ThreadPool,Pool
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pyut
from sklearn.model_selection import train_test_split


def param():
    clfparam = dict()
    clfparam['eta'] = 0.3
    clfparam['objective'] = 'multi:softmax'
    clfparam['num_class'] = 3
    #clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 1
    clfparam['max_depth'] = 6
    #clfparam['subsample'] = 0.5

    # clfparam['eval_metric'] = ['mlogloss','merror']
    #clfparam['max_delta_step'] = 1
    #clfparam['scale_pos_weight'] = 29228.0/199979
    #clfparam['updater'] = 'grow_gpu'
    return clfparam


if __name__ == '__main__':
    con, cur = pyut.poolconn()
    cur.execute('select kdj,label from kdj')
    res = cur.fetchall()
    feature = np.asarray(list(map(lambda x: x[0], res)))
    label = np.asarray(list(map(lambda x: x[1], res)))
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, random_state=42)
    TD = xgb.DMatrix(data=X_train, label=y_train)
    ED = xgb.DMatrix(data=X_test, label=y_test)
    watchlist = [(TD, 'train'), (ED, 'eval')]
    booster = xgb.train(params=param(), dtrain=TD, num_boost_round=100, evals=watchlist)
    preds = booster.predict(ED)
    pc = 0.0
    cc = 0.0
    rp = 0.0
    for i in range(0, len(preds)):
        cls = int(preds[i])
        real = int(y_test[i])
        if real == 1:
            rp += 1
        if cls == 1:
            pc += 1
            if real == 1:
                cc += 1
        print(preds[i], y_test[i])
    print(cc / pc)
    print(cc / rp)
    con.close()
