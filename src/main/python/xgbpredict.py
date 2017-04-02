import xgboost as xgb
from multiprocessing.pool import ThreadPool,Pool
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pyut
from sklearn.model_selection import train_test_split
from kdjscale import kdjscale

con, cur = pyut.poolconn()


# con.autocommit = True

def param():
    clfparam = dict()
    clfparam['eta'] = 0.01
    clfparam['objective'] = 'binary:logistic'
    #clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 1
    clfparam['max_depth'] = 9
    #clfparam['subsample'] = 0.5

    clfparam['eval_metric'] = ['error', 'error@0.7']
    #clfparam['max_delta_step'] = 1
    #clfparam['scale_pos_weight'] = 29228.0/199979
    #clfparam['updater'] = 'grow_gpu'
    return clfparam


def result(preds, label, threshold):
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
    print(threshold)
    try:
        cur.execute('INSERT INTO rate(pcr,rcr,threshold) values(%s,%s,%s)', (cc / pc, cc / rc, threshold))
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
    # fromdb()
    feature = np.load('data/xgb/feature.npy')
    label = np.load('data/xgb/label.npy')
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.33)
    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    dtest = xgb.DMatrix(data=x_test, label=y_test)
    # dtrain.save_binary('data/xgb/dtrain.matrix')
    # dtest.save_binary('data/xgb/dtest.matrix')
    return (dtrain, dtest)


if __name__ == '__main__':
    (dtrain, dtest) = datapreparation()
    cur.execute('delete from rate')
    # dtrain = xgb.DMatrix('data/xgb/dtrain.matrix')
    # dtest = xgb.DMatrix('data/xgb/dtest.matrix')
    bst = xgb.train(dtrain=dtrain, num_boost_round=100, params=param())
    preds = bst.predict(dtest)
    label = dtest.get_label()
    ran = np.arange(0, 0.73, 0.01)
    rrr = list(map(lambda x: result(preds, label, x), list(ran)))
    con.commit()
    pcr = list(map(lambda x: x[0], rrr))
    rcr = list(map(lambda x: x[1], rrr))
    plt.plot(ran, pcr, 'r', ran, rcr, 'b')
    plt.show()
    print(result2(preds, label, 0.67,0.7))
    pyut.putcon(con)