import math
from datetime import datetime
from functools import reduce
from multiprocessing.pool import Pool

import numpy as np
import psycopg2
import tensorflow as tf
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


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
        feature[i - 4] = concat(e, b, c, d, a)
        # feature[i-4] = np.concatenate((e,d,c,b,a),axis=0)
    return feature, label[4:]


def rnnpercode(code):
    try:
        feature, label = percode(code)
    except TypeError:
        return
    np.save('data/numpy/feature/' + code, feature)
    np.save('data/numpy/label/' + code, label)

def concat(d0, d1, d2, d3, d4):
    dd0 = d1 - d0
    dd1 = d2 - d1
    dd2 = d3 - d2
    dd3 = d4 - d3
    return np.concatenate((d0, dd0, d1, dd1, d2, dd2, d3, dd3, d4), axis=0)


def featureconcat(a, b):
    return np.concatenate((a, b), axis=0)


def preparernn():
    con, cur = getcon()
    cur.execute('select distinct code from raw')
    codes = [x[0] for x in cur.fetchall()]
    con.close()
    pp = Pool(8)
    pp.map(rnnpercode, codes)
    # list(map(rnnpercode,codes))


def fromdb():
    starttime = datetime.now()
    feature = None
    label = None

    FromDB = False
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
        np.save('data/rawfeature', feature)
        np.save('data/rawlabel', label)
        exit(0)
    else:
        feature = np.load('data/rawfeature.npy')
        label = np.load('data/rawlabel.npy')
    print(feature.shape)
    print(label.shape)

    if PreProcessBeforeSave:
        feature = preprocess(feature)
    np.save('data/mx/feature', feature)  # (-1,1920)
    np.save('data/mx/label', label)  # (-1,)
    endtime = datetime.now()
    print('finished save from DB, time cost: %d s' % (endtime - starttime).seconds)


def Scale(X):
    return preprocessing.StandardScaler().fit_transform(X)


def minmax(X):
    return preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(X)


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
    return X.reshape(len(X), 5, 384)


def load():
    feature = np.load('data/mx/feature.npy')
    label = np.load('data/mx/label.npy')
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, random_state=42)
    # x_train = x_train.reshape(len(x_train),5,384)
    # x_test = x_test.reshape((len(x_test),5,384))
    np.save('data/mx/x_train', x_train)
    np.save('data/mx/x_test', x_test)
    np.save('data/mx/y_train', y_train)
    np.save('data/mx/y_test', y_test)
    return x_train, x_test, y_train, y_test


def quickload():
    x_train = np.load('data/mx/x_train.npy')
    x_test = np.load('data/mx/x_test.npy')
    y_train = np.load('data/mx/y_train.npy')
    y_test = np.load('data/mx/y_test.npy')
    return x_train, x_test, y_train, y_test


def customizedfbeta(preds, dtrain):
    label = dtrain.get_label()

    threshold = 0.5

    TP = FP = TN = FN = 0.0
    for i in range(len(preds)):
        if preds[i] > threshold:
            if label[i] > threshold:
                TP += 1
            else:
                FP += 1
        else:
            if label[i] < threshold:
                TN += 1
            else:
                FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 'F1', 2 * precision * recall / (precision + recall)


def ring():
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load("clock.mp3")
    pygame.mixer.music.play()
    import time
    time.sleep(1)
    print('finished!')


def logisticlayerwithrelu(x, num_hidden, num_classes, name, with_relu=True):
    num_hidden = int(num_hidden)
    num_classes = int(num_classes)
    with tf.name_scope(name):
        W = tf.Variable(tf.random_normal((num_hidden, num_classes), stddev=1.0 / math.sqrt(num_hidden)))
        b = tf.Variable(tf.random_normal((num_classes,)))
        layer = tf.matmul(x, W) + b
        if with_relu:
            layer = tf.nn.relu(layer)
    return layer


def nextbatch(x, y, BatchSize, batch_num, shuffle=False):
    if shuffle:
        extract = np.random.randint(0, len(y), BatchSize)
        rx = np.zeros((BatchSize, len(x[0])), dtype=np.float32)
        ry = np.zeros((BatchSize,), dtype=np.float32)
        for i in range(BatchSize):
            rx[i] = x[extract[i]]
            ry[i] = y[extract[i]]
            return rx, ry
    else:
        return x[BatchSize * batch_num:BatchSize * (batch_num + 1)], y[
                                                                     BatchSize * batch_num:BatchSize * (batch_num + 1)]


# noinspection PyTypeChecker
def tfplayground():
    x_train, x_test, y_train, y_test = quickload()

    tf_log_path = 'data/tf_log/' + str(datetime.now())[0:19].replace(':', '-')

    epochs = 60

    BatchSize = 256
    FeatureLength = 3456
    num_classes = 2
    LearningRate = 0.01
    num_hidden_layers = 6
    num_hidden_units = 1728

    with tf.name_scope('feature'):
        feature = tf.placeholder(tf.float32, (BatchSize, FeatureLength), name='input_feature')

    with tf.name_scope('label'):
        rawlabel = tf.to_int32(tf.placeholder(tf.float32, (BatchSize,), name='rawlabel'))
        label = tf.one_hot(rawlabel, num_classes, name='onehot')

    net = logisticlayerwithrelu(feature, FeatureLength, num_hidden_units, 'layer1')

    for i in range(2, num_hidden_layers):
        net = logisticlayerwithrelu(net, num_hidden_units, num_hidden_units, 'layer' + str(i))

    net = logisticlayerwithrelu(net, num_hidden_units, num_classes, 'layer' + str(num_hidden_layers), with_relu=True)

    with tf.name_scope('softmax output'):
        out = tf.nn.softmax(net)

    with tf.name_scope('metrics'):
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(out, 1), tf.argmax(label, 1))))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=label))

    with tf.name_scope('optimizer'):
        opt = tf.train.GradientDescentOptimizer(learning_rate=LearningRate).minimize(loss)

    summary_loss = tf.summary.scalar('loss-by-batch', loss)
    summary_train_acc = tf.summary.scalar('train-accuracy-by-batch', accuracy)
    # summary_val_acc = tf.summary.scalar('validation-acc',accuracy)
    SummaryWriter = tf.summary.FileWriter(tf_log_path, graph=tf.get_default_graph())

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        ring()
        # train
        trainbatchsize = int(len(x_train) / BatchSize)
        valbatchsize = int(len(x_test) / BatchSize)
        for epoch in range(epochs):
            for i in range(trainbatchsize):
                batchx, batchy = nextbatch(x_train, y_train, BatchSize, i, shuffle=True)
                _, acc_summary, loss_summary = sess.run([opt, summary_train_acc, summary_loss], feed_dict={
                    feature: batchx,
                    rawlabel: batchy
                })
                SummaryWriter.add_summary(acc_summary, epoch * trainbatchsize + i)
                SummaryWriter.add_summary(loss_summary, epoch * trainbatchsize + i)
            # log metrics to tensorboard

            batchaccuracy = []
            for i in range(trainbatchsize):
                batchx, batchy = nextbatch(x_train, y_train, BatchSize, i)
                log_acc = sess.run(accuracy, feed_dict={
                    feature: batchx,
                    rawlabel: batchy
                })
                batchaccuracy.append(log_acc)
            train_acc = np.mean(batchaccuracy)
            acc_summary = tf.Summary()
            acc_summary.value.add(tag='train-accuracy', simple_value=train_acc)
            SummaryWriter.add_summary(acc_summary, epoch)

            batchaccuracy = []
            for i in range(valbatchsize):
                batchx, batchy = nextbatch(x_train, y_train, BatchSize, i)
                log_acc = sess.run(accuracy, feed_dict={
                    feature: batchx,
                    rawlabel: batchy
                })
                batchaccuracy.append(log_acc)
            val_acc = np.mean(batchaccuracy)
            acc_summary = tf.Summary()
            acc_summary.value.add(tag='val-accuracy', simple_value=val_acc)
            SummaryWriter.add_summary(acc_summary, epoch)
            print('epoch %d completed at %s\n    train-acc: %.4f\n    val-acc: %.4f' % (
                epoch, str(datetime.now())[0:19].replace(':', '-'), float(train_acc), float(val_acc)))

    ring()


if __name__ == '__main__':
    # load()
    # ring()
    # tfplayground()
    preparernn()
