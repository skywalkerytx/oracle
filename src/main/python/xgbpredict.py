import xgboost as xgb
from multiprocessing.pool import ThreadPool,Pool
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pyut
from sklearn.model_selection import train_test_split
from kdjscale import kdjscale


def param():
    clfparam = dict()
    clfparam['eta'] = 0.01
    clfparam['objective'] = 'multi:softmax'
    clfparam['num_class'] = 3
    #clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 1
    clfparam['max_depth'] = 6
    #clfparam['subsample'] = 0.5

    # clfparam['eval_metric'] = ['myerror1','myerror2']
    #clfparam['max_delta_step'] = 1
    #clfparam['scale_pos_weight'] = 29228.0/199979
    #clfparam['updater'] = 'grow_gpu'
    return clfparam


def CM1(preds, dtrain):
    y_test = dtrain.get_label()
    pc = 0.0
    cc = 0.0
    for i in range(0, len(y_test)):
        cls = int(preds[i])
        real = int(y_test[i])
        if cls == 1:
            pc += 1
            if real == 1:
                cc += 1
    return 'myerror1', cc / pc


def CM2(preds, dtrain):
    y_test = dtrain.get_label()
    pc = 0.0
    cc = 0.0
    for i in range(0, len(y_test)):
        cls = int(preds[i])
        real = int(y_test[i])
        if real == 1:
            pc += 1
            if cls == 1:
                cc += 1
    return 'myerror2', cc / pc


def predictbycode(code):
    con, cur = pyut.poolconn()
    cur.execute('select kdj,label from kdj where code = %s', (code,))
    res = cur.fetchall()
    if len(res) < 5:
        return 0, 0
    feature = np.asarray(list(map(lambda x: x[0], res)))
    label = np.asarray(list(map(lambda x: x[1], res)))
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, random_state=42)
    train_weight = np.ones((len(y_train), 1))
    for i in range(0, len(y_train)):
        if int(y_train[i]) == 1:
            train_weight[i] = 10
        if int(y_train[i]) == 2:
            train_weight[i] = 5
    TD = xgb.DMatrix(data=X_train, label=y_train, weight=train_weight)
    ED = xgb.DMatrix(data=X_test, label=y_test)
    watchlist = [(TD, 'train'), (ED, 'eval')]
    booster = xgb.train(params=param(), dtrain=TD, num_boost_round=500)  # , evals=watchlist,maximize=True)
    booster.save_model('data/xgb/' + code + '.model')
    preds = booster.predict(ED)

    pc = 0.0
    cc = 0.0
    for i in range(0, len(preds)):
        if int(preds[i]) == 1:
            pc += 1
            if y_test[i] == 1:
                cc += 1
    if pc != 0:
        print(cc / pc)
    pyut.putcon(con)
    return cc, pc

if __name__ == '__main__':
    # kdjscale()
    '''
    codes = pyut.getcode()
    cc = 0.0
    pc = 0.0
    for code in codes:
        a,b = predictbycode(code)
        cc+=a
        pc+=b

    print(cc/pc)
    print(cc,pc)
    '''
    ccdic = dict()
    pcdic = dict()
    ratedic = dict()
    import os

    models = os.listdir('data/xgb')
    con, cur = pyut.poolconn()
    cur.execute('select kdj,label from kdj')
    res = cur.fetchall()
    feature = np.asarray(list(map(lambda x: x[0], res)))
    label = np.asarray(list(map(lambda x: x[1], res)))
    ED = xgb.DMatrix(data=feature, label=label)
    for model in models:
        booster = xgb.Booster(model_file='data/xgb/' + model)
        preds = booster.predict(ED)
        pc = 0.0
        cc = 0.0
        for i in range(0, len(preds)):
            if int(preds[i]) == 1:
                pc += 1
                if label[i] == 1:
                    cc += 1
        ccdic[model] = cc
        pcdic[model] = pc
        if pc != 0:
            ratedic[model] = cc / pc
            print(
                '''%s:
                    %f''' % (model, cc / pc))
    print(sorted(ratedic.values()))

    pyut.putcon(con)
