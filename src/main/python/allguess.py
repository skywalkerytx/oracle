import math
from datetime import datetime
from functools import reduce
from multiprocessing.pool import Pool
import os
import pygame
import time

import numpy as np
import psycopg2
import tensorflow as tf

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


n_steps = 5

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
    rawfeature = preprocess(rawfeature)
    feature = np.zeros((len(rawfeature) - 4, 5 * len(rawfeature[0])))
    for i in range(4, len(rawfeature)):
        a = rawfeature[i]
        b = rawfeature[i - 1]
        c = rawfeature[i - 2]
        d = rawfeature[i - 3]
        e = rawfeature[i - 4]
        #feature[i - 4] = concat(e, b, c, d, a)
        feature[i-4] = np.concatenate((e,d,c,b,a),axis=0)
    return feature, label[4:]


def rnnpercode(code):
    try:
        feature, label = percode(code)
    except TypeError:
        return
    if len(feature) < 142:
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

def pca(X):
    pca = PCA()
    return PCA.fit_transform(X)

def preprocess(X):
    X = minmax(X)
    X = pca(X)
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

def ring():
    pygame.mixer.init()
    pygame.mixer.music.load("clock.mp3")
    pygame.mixer.music.play()
    time.sleep(2)
    print('finished!')


def logisticlayerwithrelu(x, num_hidden, num_classes, name, with_relu=True,with_dropout = False,keep_prob = 0.9):
    num_hidden = int(num_hidden)
    num_classes = int(num_classes)
    with tf.name_scope(name):
        W = tf.Variable(tf.random_normal((num_hidden, num_classes), stddev=0.01))
        b = tf.Variable(tf.random_normal((num_classes,)))
        layer = tf.matmul(x, W) + b
        if with_relu:
            layer = tf.nn.relu(layer)
        if with_dropout:
            layer = tf.nn.dropout(layer,keep_prob=keep_prob)
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
def network(feature,label,BatchSize,n_steps,n_inputs,n_hidden,n_classes,num_lstm_cell):

    net = logisticlayerwithrelu(feature,n_hidden,n_hidden,name='logistic-with-relu')

    with tf.name_scope('lstm'):
        net = tf.split(net, n_steps, 0)  # (TimeStep*BatchSize,Features) => [(BatchSize,Features)]*TimeStep

        lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0) for _ in range(num_lstm_cell)])
        output, state = tf.contrib.rnn.static_rnn(lstm, net, dtype=tf.float32)
        net = output[-1]

    net = logisticlayerwithrelu(net,n_hidden,n_classes,name='final-logisitc',with_relu=False)
    net = tf.nn.softmax(net)
    return net

def getcode():
    for root, dirs, files in os.walk('data/numpy/feature/'):
        return [file[0:9] for file in files]

def tfplayground():

    tf_log_path = 'data/tf_log/' + str(datetime.now())[0:19].replace(':', '-')

    epochs = 60
    BatchSize = 145-n_steps


    n_inputs = 384
    n_hidden = 384
    n_classes = 2
    max_clip = 10

    LearningRate = 1e-4

    with tf.name_scope('feature'):
        rawfeature = tf.placeholder(tf.float32, (BatchSize, n_steps*n_inputs), name='rawinput')  # (BatchSize,TimeStep*Features)
        feature = tf.reshape(rawfeature, (BatchSize, n_steps, n_inputs))  # (BatchSize,TimeStep,Features)
        feature = tf.transpose(feature, (1, 0, 2))  # (TimeStep,BatchSize,Features)
        feature = tf.reshape(feature, (-1, n_hidden)) #(TimeStep*BatchSize,Features)


    with tf.name_scope('label'):
        rawlabel = tf.to_int32(tf.placeholder(tf.float32, (BatchSize,), name='rawlabel'))
        label = tf.one_hot(rawlabel, n_classes, name='onehot')

    net = network(feature,label,BatchSize,n_steps,n_inputs,n_hidden,n_classes,num_lstm_cell=5)

    with tf.name_scope('loss-function'):
        loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=net))

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(LearningRate)
        # Op to calculate every variable gradient
        grads = optimizer.compute_gradients(loss_func)
        #grads = [(tf.clip_by_norm(grad,10), var) for grad,var in grads]
        clipped = [(tf.clip_by_value(grad,-max_clip,max_clip), var) for grad, var in grads]
        # Op to update all variables according to their gradient
        apply_grads = optimizer.apply_gradients(grads_and_vars=clipped)


    with tf.name_scope('metrics'):
        accuracy = tf.equal(tf.arg_max(net, 1), tf.arg_max(label, 1))
        accuracy = tf.reduce_mean(tf.to_float(accuracy))

    with tf.name_scope('log-by-batch'):
        log_accuracy = tf.summary.scalar('accuracy-by-batch',accuracy)
        log_loss = tf.summary.scalar('loss-by-batch',loss_func)

    with tf.name_scope('hista'):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var)
        for grad,var in clipped:
            tf.summary.histogram(var.name+'/gradient',grad)

    merged_summary_op = tf.summary.merge_all()

    codes = getcode()
    totalbatch = len(codes)
    SummaryWriter = tf.summary.FileWriter(tf_log_path,graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Acc = 0.0
        for epoch in range(epochs):
            Acc = 0.0
            Loss = 0.0
            for i in range(totalbatch):
                x, y = rnnbatch(codes[i], BatchSize)
                _,merged_sum,batchacc,batchloss = sess.run([apply_grads,merged_summary_op,accuracy,loss_func],
                         feed_dict = {
                             rawfeature:x,
                             rawlabel:y
                         }
                         )
                Acc += batchacc
                Loss += batchloss
                SummaryWriter.add_summary(merged_sum,epoch*totalbatch+i)
                #SummaryWriter.add_summary(acc_sum,epoch*totalbatch+i)
                #SummaryWriter.add_summary(loss_sum,epoch*totalbatch+i)

            #summary per epoch
            Acc /=totalbatch
            Loss /= totalbatch
            with tf.name_scope('log-by-epoch'):
                acc_sum = tf.Summary()
                loss_sum = tf.Summary()
                acc_sum.value.add(tag='train-accuracy', simple_value=Acc)
                loss_sum.value.add(tag='train-loss', simple_value=Loss)
                SummaryWriter.add_summary(acc_sum, epoch)
                SummaryWriter.add_summary(loss_sum, epoch)
            print('epoch %d finished at %s with Accuracy: %.4f'%(epoch,str(datetime.now())[11:19],Acc))
            #time.sleep(5) #my graphic card's fan speed is fucked when training





def rnnbatch(code,BatchSize):
    x = np.load('data/numpy/feature/'+code+'.npy')
    size = len(x)
    y = np.load('data/numpy/label/'+code+'.npy')
    pick = np.random.randint(0,len(x),(BatchSize,))
    x = np.asarray([x[i] for i in pick])
    y = np.asarray([y[i] for i in pick])
    return x,y


def rnnstats():
    import os

    for root, dirs, files in os.walk('data/numpy/feature/'):
        lens = [len(np.load('data/numpy/feature/' + file)) for file in files]
        print(np.min(lens))
        print(np.mean(lens))
        print(np.max(lens))
        feature = np.load('data/numpy/feature/sh600000 .npy')
        print(feature)
        print(feature.shape)



if __name__ == '__main__':
    # load()
    # ring()
    #preparernn()
    #rnnstats()
    tfplayground()

