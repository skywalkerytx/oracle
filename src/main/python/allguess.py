import psycopg2
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import mxnet as mx
from mxnet.symbol import Variable as var
import pyut


def param():
    clfparam = dict()
    clfparam['eta'] = 0.01
    clfparam['objective'] = 'binary:logistic'
    # clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 0
    clfparam['max_depth'] = 12
    # clfparam['subsample'] = 0.5

    clfparam['eval_metric'] = ['error', 'error@0.7']
    # clfparam['max_delta_step'] = 1
    # clfparam['scale_pos_weight'] = 29228.0/199979
    # clfparam['updater'] = 'grow_gpu'
    return clfparam


con, cur = pyut.poolconn()


def save(a, b, c, d):
    np.save('data/mx/x_train', a)
    np.save('data/mx/x_test', b)
    np.save('data/mx/y_train', c)
    np.save('data/mx/y_test', d)


def load():
    a = mx.nd.array(np.load('data/mx/x_train.npy'), mx.gpu(0), dtype=np.float16)
    b = mx.nd.array(np.load('data/mx/x_test.npy'), mx.gpu(0), dtype=np.float16)
    c = mx.nd.array(np.load('data/mx/y_train.npy'), mx.gpu(0), dtype=np.float16)
    d = mx.nd.array(np.load('data/mx/y_test.npy'), mx.gpu(0), dtype=np.float16)
    return (a, b, c, d)


def softmax(label, num_class=2):
    res = np.zeros(num_class)
    if label > 0.5:
        res[1] = 1
    else:
        res[0] = 1
    return res


def fromdb():
    cur.execute("""
select 
vector.vector,
label.vector[1] 
from vector inner join label on vector.code = label.code and vector.date = label.date order by vector.code asc,vector.date asc
""")
    res = cur.fetchall()
    feature = np.asarray([row[0] for row in res])
    label = np.asarray([row[1] for row in res])
    del res
    # label = np.asarray(list(map(lambda x:x[1],res)))
    # feature = feature.reshape((len(res),384))
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    feature = preprocessing.MinMaxScaler().fit_transform(feature)
    feature = PCA().fit_transform(feature)
    feature = preprocessing.MinMaxScaler().fit_transform(feature)
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, random_state=42)
    # y_train = np.asarray([softmax(x,num_class=2) for x in y_train])
    # y_test = np.asarray([softmax(x,num_class=2) for x in y_test])
    save(x_train, x_test, y_train, y_test)


def network():
    data = var('data')
    label = var('softmax_label')
    fullc = mx.sym.FullyConnected(data=data, num_hidden=1)
    loss = mx.sym.SoftmaxOutput(data=data, label=label)
    mod = mx.mod.Module(loss)
    return mod


# fromdb()

x_train, x_test, y_train, y_test = load()

BatchSize = 50

train_iter = mx.io.NDArrayIter(
    data={'data': x_train},
    label={'softmax_label': y_train},
    batch_size=BatchSize
)
# mx.io.NDArrayIter(data=x_train,label=y_train)

print(train_iter.provide_data, train_iter.provide_label)

network().fit(train_iter, num_epoch=2)
