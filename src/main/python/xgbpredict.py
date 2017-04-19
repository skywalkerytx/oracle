import matplotlib.pyplot as plt
import numpy as np
import pyut
import xgboost as xgb
from itertools import repeat
from kdjscale import kdjscale
from multiprocessing.pool import Pool
from sklearn.model_selection import train_test_split

con, cur = pyut.poolconn()


# con.autocommit = True





def result(para):
    preds, label, threshold = para
    pc = 0.0
    cc = 0.0
    rc = 0.0
    for i in range(0, len(preds)):
        if label[i] >= 0.5:
            rc += 1
        if preds[i] >= threshold:
            pc += 1
            if label[i] >= 0.5:
                cc += 1
    # print(threshold)
    try:
        # cur.execute('INSERT INTO rate(pcr,rcr,threshold) values(%s,%s,%s)', (cc / pc, cc / rc, threshold))
        return [cc / pc, cc / rc]
    except ZeroDivisionError:
        return result(preds, label, threshold=threshold - 0.01)


def result2(preds, label, threshold1, threshold2):
    pc = 0.0
    cc = 0.0
    rc = 0.0
    for i in range(0, len(preds)):
        if label[i] >= 0.5:
            rc += 1
        if preds[i] >= 0.65 and preds[i] < 0.7:
            pc += 1
            if label[i] >= 0.5:
                cc += 1
    print(threshold1, threshold2)
    # cur.execute('INSERT INTO rate(pcr,rcr,threshold) values(%s,%s,%s)',(cc/pc,cc/rc,threshold))
    return [cc / pc, cc / rc]


def fromdb():
    kdjscale()
    cur.execute('select kdj,label from kdj where label is not null')
    res = cur.fetchall()
    feature = np.asarray(list(map(lambda x: x[0], res)))
    label = np.asarray(list(map(lambda x: x[1], res)))
    np.save('data/xgb/feature', feature)
    np.save('data/xgb/label', label)


def datapreparation():
    fromdb()
    feature = np.load('data/xgb/feature.npy')
    label = np.load('data/xgb/label.npy')
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, random_state=42)
    # dtrain = xgb.DMatrix(data=x_train, label=y_train)  # ,weight=weight)
    # dtest = xgb.DMatrix(data=x_test, label=y_test)
    # dtrain.save_binary('data/xgb/dtrain.matrix')
    # dtest.save_binary('data/xgb/dtest.matrix')
    return (x_train, x_test, y_train, y_test)


def param():
    clfparam = dict()
    clfparam['eta'] = 0.01
    clfparam['objective'] = 'binary:logistic'
    # clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 0
    clfparam['max_depth'] = 9
    # clfparam['subsample'] = 0.5

    clfparam['eval_metric'] = ['error', 'error@0.7']
    # clfparam['max_delta_step'] = 1
    # clfparam['scale_pos_weight'] = 29228.0/199979
    # clfparam['updater'] = 'grow_gpu'
    return clfparam


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = datapreparation()
    cur.execute('delete from rate')
    # dtrain = xgb.DMatrix('data/xgb/dtrain.matrix')
    # dtest = xgb.DMatrix('data/xgb/dtest.matrix')
    count = 0
    tp = Pool(8)
    rounds = 100
    bst = xgb.XGBClassifier(max_depth=81, learning_rate=0.01, n_estimators=100, silent=False,
                            objective='binary:logistic',
                            nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                            colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0,
                            missing=None)
    bst.fit(x_train, y_train)
    # xgb.plot_importance(bst)
    plt.bar(range(len(bst.feature_importances_)), bst.feature_importances_)
    plt.show()
    exit(0)
    preds = bst.predict(x_test)
    label = y_test
    ran = np.arange(0, 0.8, 0.01)
    para = list(zip(repeat(preds), repeat(label), list(ran)))
    rrr = tp.map(result, para)
    con.commit()
    pcr = list(map(lambda x: x[0], rrr))
    rcr = list(map(lambda x: x[1], rrr))
    count += 1
    # plt.plot(ran, pcr, 'r', ran, rcr, 'b')
    # plt.show()
    # print(result2(preds, label, 0.67,0.7))

    pyut.putcon(con)
