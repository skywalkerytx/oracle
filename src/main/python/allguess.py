import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from multiprocessing.pool import Pool



def getcon():
    conn = psycopg2.connect(database='nova', user='nova', password='emeth')
    cur = conn.cursor()
    return conn, cur


def percode(code):
    con, cur = getcon()

    cur.execute('''
    SELECT
        vector.vector,
        label.vector[1]
    FROM
        vector
    INNER JOIN
        label
    ON
        vector.code = label.code
    AND
        vector.date = label.date
    WHERE
        vector.code = %s
    ORDER BY
        label.date
            ASC
    ''', (code,))
    res = cur.fetchall()
    con.close()
    if len(res) < 5:
        return None
    rawfeature = np.zeros((len(res), len(res[0][0])))
    label = np.zeros(len(res))
    for i in range(len(res)):
        rawfeature[i] = res[i][0]
        label[i] = res[i][1]
    feature = np.zeros((len(rawfeature) - 4, 5 * len(rawfeature[0])))
    for i in range(4, len(rawfeature)):
        a = rawfeature[i]
        b = rawfeature[i - 1]
        c = rawfeature[i - 2]
        d = rawfeature[i - 3]
        e = rawfeature[i - 4]
        #feature[i - 4] = concat(a, b, c, d, e)
        feature[i-4] = np.concatenate((e,d,c,b,a),axis=0)
    return (feature, label[4:])


def concat(d0, d1, d2, d3, d4):
    dd0 = d1 - d0
    dd1 = d2 - d1
    dd2 = d3 - d2
    dd3 = d4 - d3
    return np.concatenate((d0, dd0, d1, dd1, d2, dd2, d3, dd3, d4), axis=0)


def featureconcat(a, b):
    return np.concatenate((a, b), axis=0)

from datetime import datetime

def fromDB():
    feature = None
    label = None

    FromDB= False
    PreProcessBeforeSave = True

    if FromDB:
        starttime = datetime.now()
        con, cur = getcon()
        cur.execute('select distinct code from raw')
        codes = [x[0] for x in cur.fetchall()]
        con.close()
        pp = Pool(8)
        feature, label = reduce(lambda a, b: (featureconcat(a[0], b[0]), featureconcat(a[1], b[1])),
                                filter(lambda x: x is not None, pp.map(percode, codes)))
        np.save('data/rawfeature',feature)
        np.save('data/rawlabel',label)
    else:
        feature = np.load('data/rawfeature.npy')
        label = np.load('data/rawlabel.npy')
    print(feature.shape)
    print(label.shape)

    if PreProcessBeforeSave:
        feature = preprocess(feature)
    np.save('data/mx/feature', feature) #(-1,1920)
    np.save('data/mx/label', label) #(-1,)
    endtime = datetime.now()
    print('finished save from DB, time cost: %d s'%(endtime-starttime).seconds)


def Scale(X):
    return preprocessing.StandardScaler().fit_transform(X)

def minmax(X):
    return preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(X)

def Normalize(X):
    normalizer = preprocessing.Normalizer()
    return normalizer.fit_transform(X)

def preprocess(X):
    X = minmax(X)
    pca = PCA()
    pca.fit(X)
    X = pca.transform(X)
    X = Normalize(X)
    return X

def totimestep(X):
    return X.reshape(len(X),5,384)

def load():
    feature = np.load('data/mx/feature.npy')
    label = np.load('data/mx/label.npy')
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, random_state=42)
    #x_train = x_train.reshape(len(x_train),5,384)
    #x_test = x_test.reshape((len(x_test),5,384))
    return x_train, x_test, y_train, y_test



if __name__ == '__main__':
    #fromDB()
    x_train, x_test, y_train, y_test = load()
    print('here we go')
    import nets
    import mxnet as mx

    BatchSize = 400
    labelname = 'label'
    train_iter = mx.io.NDArrayIter(
        data={'data':x_train},
        label={labelname:y_train},
        batch_size=BatchSize
    )
    val_iter = mx.io.NDArrayIter(
        data={'data':x_test},
        label = {labelname:y_test},
        batch_size=BatchSize
    )
    net = nets.LinearRegression(2)
    model = mx.mod.Module(
        symbol=net,
        context=mx.gpu(0),
        data_names=['data'],
        label_names = [labelname]
    )
    import logging
    logging.basicConfig(level=logging.INFO)
    epochs = 10
    print('training')
    model.fit(
        train_iter,
        optimizer='sgd',
        eval_metric=['acc','f1'],
        #eval_data=val_iter,
        num_epoch=epochs
    )
    val_iter.reset()
    P = model.predict(val_iter).asnumpy()
    print(P)
    preds = np.zeros(len(P))
    for i in range(len(P)):
        if P[i][0]>0.5:
            preds[i]=1

    for i in preds:
        if i:
            print(i)
