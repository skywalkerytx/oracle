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
    feature = np.zeros((len(rawfeature) - 4, 9 * len(rawfeature[0])))
    for i in range(4, len(rawfeature)):
        a = rawfeature[i]
        b = rawfeature[i - 1]
        c = rawfeature[i - 2]
        d = rawfeature[i - 3]
        e = rawfeature[i - 4]
        feature[i - 4] = concat(a, b, c, d, e)
    return (feature, label[4:])


def concat(d0, d1, d2, d3, d4):
    dd0 = d1 - d0
    dd1 = d2 - d1
    dd2 = d3 - d2
    dd3 = d4 - d3
    return np.concatenate((d0, dd0, d1, dd1, d2, dd2, d3, dd3, d4), axis=0)


def featureconcat(a, b):
    return np.concatenate((a, b), axis=0)


def fromDB():
    con, cur = getcon()
    cur.execute('select distinct code from raw')
    codes = [x[0] for x in cur.fetchall()]
    con.close()
    pp = Pool(8)
    feature, label = reduce(lambda a, b: (featureconcat(a[0], b[0]), featureconcat(a[1], b[1])),
                            filter(lambda x: x is not None, pp.map(percode, codes)))
    print(feature.shape)
    print(label.shape)
    PreProcessBeforeSave = True
    if PreProcessBeforeSave:
        feature = preprocess(feature)
    np.save('data/mx/feature', feature)
    np.save('data/mx/label', label)


def Scale(X):
    return preprocessing.StandardScaler().fit(X).transform(X)


def preprocess(X):
    X = Scale(X)
    pca = PCA()
    pca.fit(X)
    X = pca.transform(X)
    X = Scale(X)
    return X


def load():
    feature = np.load('data/mx/feature.npy')
    label = np.load('data/mx/label.npy')
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    fromDB()
    input('finished save from DB')
    a, b, c, d = load()
    input('...')
