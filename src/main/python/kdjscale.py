import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split
import psycopg2
import tensorflow as tf
from functools import reduce
from datetime import datetime
import time

n_steps = 5
n_inputs = 9 # k,d,j,kdjcross,macddif,macddea,macdmacd,macdcross,cross=cross 9 total
n_hidden = 9
n_classes = 2

BatchSize = 256
LearningRate = 0.001
Epochs = 5000

tfdtype = tf.float32


con = psycopg2.connect(database = 'nova',user = 'nova',password = 'nova')

cur = con.cursor()

def preprocess(X):
    X = preprocessing.minmax_scale(X,feature_range=(-1,1))
    #pca = decomposition.PCA()
    #X = pca.fit_transform(X)
    #X = preprocessing.minmax_scale(X, feature_range=(-1, 1))
    return X

def fromdb():
    cur.execute('''
    SELECT
      k,
      d,
      j,
      CASE WHEN kdjcross = '金叉' THEN 1 ELSE 0 END AS kdjcross_bool,
      macddif,
      macddea,
      macdmacd,
      CASE WHEN macdcross = '金叉' THEN 1 ELSE 0 END AS macdcross_bool,
      CASE WHEN macdcross = kdjcross AND kdjcross = '金叉'  THEN 1 ELSE 0 END AS Resonance,
      label.vector[1]
    FROM raw
      INNER JOIN label
      ON raw.code=label.code AND raw.date=label.date
    ''')

    rawdata = cur.fetchall()

    feature = [x[0:9] for x in rawdata]
    label = [x[-1] for x in rawdata]

    feature = preprocess(np.asarray(feature,dtype=np.float32))
    label = np.asarray(label,dtype=np.int32)

    global n_inputs,n_hidden

    n_inputs = int(feature.shape[1])
    n_hidden = n_inputs

    print(feature.shape)
    print(label.shape)

    x_train, x_val, y_train, y_val = train_test_split(feature, label, test_size=0.3, random_state=42)
    np.save('data/kdjonly/x_train',x_train)
    np.save('data/kdjonly/x_val',x_val)
    np.save('data/kdjonly/y_train',y_train)
    np.save('data/kdjonly/y_val',y_val)

def rnnpercode(code):
    cur.execute('''
                SELECT
                  op,
                  mx,
                  k,
                  d,
                  j,
                  CASE WHEN kdjcross = '金叉' THEN 1 ELSE 0 END AS kdjcross_bool,
                  macddif,
                  macddea,
                  macdmacd,
                  CASE WHEN macdcross = '金叉' THEN 1 ELSE 0 END AS macdcross_bool,
                  CASE WHEN macdcross = kdjcross AND kdjcross = '金叉'  THEN 1 ELSE 0 END AS Resonance,
                  label.vector[1]
                FROM raw
                  INNER JOIN label
                  ON raw.code=label.code AND raw.date=label.date
                WHERE 
                  raw.code = %s
                ORDER BY 
                  raw.date ASC 
                ''', (code,))
    rawdata = cur.fetchall()

    rawfeature = [x[0:9] for x in rawdata]
    label = [x[-1] for x in rawdata]

    rawfeature = preprocess(np.asarray(rawfeature, dtype=np.float32))
    rawlabel = np.asarray(label, dtype=np.int32)
    feature = np.zeros((len(rawfeature)-n_steps+1,n_inputs),dtype=np.float32)
    label = np.zeros((len(rawlabel)-n_steps+1,),dtype=np.int32)

    for i in range(len(rawlabel)-n_steps+1):
        feature[i] = np.asarray([rawfeature[i+j] for j in range(n_steps)]).reshape((n_inputs,))
        label[i] = rawlabel[i+n_steps-1]
    return feature,label

def rnnfromdb():
    cur.execute('''
    SELECT t.code
    FROM (    SELECT
      raw.code,count(1) as cnt
    FROM raw
      INNER JOIN label
      ON raw.code=label.code AND raw.date=label.date
    GROUP BY raw.code
    order by cnt ASC) t
    where t.cnt>=%s
''',(n_steps,))
    codes = [x[0] for x in cur.fetchall()]
    global n_inputs
    global n_hidden

    n_inputs = n_steps * n_inputs
    n_hidden=n_inputs
    rawdata = [rnnpercode(code) for code in codes]
    s = 0
    for i in rawdata:
        s+=len(i)
    print(s)
    feature,label = rawdata[0]

    for i in range(1,len(rawdata)):
        linefeature,linelabel = rawdata[i]
        feature = np.concatenate((feature,linefeature),axis=0)
        label = np.concatenate((label,linelabel),axis=0)
    #feature,label = reduce(lambda x,y:(np.concatenate((x[0],y[0]),axis=0),np.concatenate((x[1],y[1]),axis=0)),rawdata)

    print(feature.shape)
    print(label.shape)

    x_train, x_val, y_train, y_val = train_test_split(feature, label, test_size=0.3, random_state=42)
    np.save('data/kdjonly/x_train', x_train)
    np.save('data/kdjonly/x_val', x_val)
    np.save('data/kdjonly/y_train', y_train)
    np.save('data/kdjonly/y_val', y_val)
    con.close()


def load():
    x_train = np.load('data/kdjonly/x_train.npy')
    x_val = np.load('data/kdjonly/x_val.npy')
    y_train = np.load('data/kdjonly/y_train.npy')
    y_val = np.load('data/kdjonly/y_val.npy')
    global n_inputs
    global n_hidden
    n_hidden = n_inputs = len(x_train[0])
    return x_train,x_val,y_train,y_val

from tensorflow import placeholder,Variable,name_scope

def logisticlayer(X,n_hidden,n_classes,name, with_relu=True,with_dropout = False,keep_prob = 0.9):
    with name_scope(name):
        W = Variable(initial_value=tf.random_normal((n_hidden,n_classes),stddev=0.01))
        b = Variable(initial_value=tf.random_normal((n_classes,)))
        layer = tf.matmul(X,W)+b
        if with_dropout:
            layer = tf.nn.dropout(layer,keep_prob=keep_prob)
        if with_relu:
            layer = tf.nn.relu(layer)
    return layer

def MLP(X,n_hidden,n_layer,with_dropout = False,keep_prob = 0.75):
    net = X
    for i in range(n_layer-1):
        net = logisticlayer(net,n_hidden,n_hidden,'logistic-'+str(i),with_relu=True)
        #
    if with_dropout and n_layer>=1:
        net = logisticlayer(net, n_hidden, n_hidden, 'logistic-' + str(i), with_relu=True, with_dropout=True,
                            keep_prob=keep_prob)
    return net

def RNN(X,n_steps,n_hidden,n_lstm):
    cell_list = []
    for _ in range(n_lstm):
        newcell = tf.contrib.rnn.LSTMCell(num_units=n_hidden, forget_bias=1.0, activation=tf.nn.elu)
        cell_list.append(newcell)
    net = tf.contrib.rnn.MultiRNNCell(cell_list)
    output,state = tf.contrib.rnn.static_rnn(net, X, dtype=tf.float32)
    net = output[-1]
    net = tf.nn.dropout(net, keep_prob=0.9)
    return net


def tftrain():

    x_train, x_val, y_train, y_val = load()

    with name_scope('feature'):
        feature = placeholder(dtype=tfdtype,shape=(BatchSize,n_inputs))

    with name_scope('label'):
        rawlabel = placeholder(dtype=tf.int32,shape=(BatchSize,))
        label = tf.one_hot(rawlabel, n_classes, name='onehot')

    with name_scope('MLP'):
        net = MLP(feature,n_hidden,n_layer=n_steps,with_dropout=False,keep_prob=0.75)

    rnn_hidden = int(n_inputs/n_steps)

    with name_scope('transform-for-RNN'): #net.shape = (BatchSize, n_inputs= n_steps * n_hidden)
        net = tf.reshape(net,(BatchSize,n_steps,rnn_hidden))
        net = tf.transpose(net,(1,0,2)) # n_steps, BatchSize, n_hidden
        net = tf.reshape(net,(-1,rnn_hidden)) # n_steps * BatchSize, n_hidden
        net = tf.split(net,n_steps,0) # n_steps, BatchSize, n_hidden

    with name_scope('RNN'):
        net = RNN(net,n_steps,rnn_hidden,2)

    with name_scope('outputlr'):
        #net = tf.nn.dropout(net,keep_prob=0.75)
        net = logisticlayer(net,rnn_hidden,n_classes,with_relu=False,name='output')
        net = tf.nn.softmax(net)

    with name_scope('loss'):
        #loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=net))

        #loss_func = tf.losses.mean_squared_error(labels=label,predictions=net)
        #loss_func = tf.losses.softmax_cross_entropy(onehot_labels=label,logits=net)
        loss_func = tf.losses.log_loss(labels=label,predictions=net)

    with name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(LearningRate)
        #optimizer = tf.train.AdamOptimizer(epsilon=0.01)
        grads = optimizer.compute_gradients(loss_func)
        apply_grads = optimizer.apply_gradients(grads)

    with name_scope('early-optimizer'):
        jumper = tf.train.GradientDescentOptimizer(0.01).minimize(loss_func)

    with tf.name_scope('metrics'):
        accuracy = tf.equal(tf.arg_max(net, 1), tf.arg_max(label, 1))
        accuracy = tf.reduce_mean(tf.to_float(accuracy))

    with tf.name_scope('log-by-batch'):
        log_accuracy = tf.summary.scalar('accuracy',accuracy)
        log_loss = tf.summary.scalar('loss',loss_func)

    '''
    with tf.name_scope('weights'):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var)

    with tf.name_scope('grads'):
        for grad,var in grads:
            tf.summary.histogram(var.name+'/gradient',grad)
    '''
    merged_summary_op = tf.summary.merge_all()

    ModelPath = str(datetime.now())[0:19].replace(':', '-')
    tf_log_path = 'data/tf_log/' + ModelPath
    SummaryWriter = tf.summary.FileWriter(tf_log_path, graph=tf.get_default_graph())

    trainbatch = int(len(x_train)/BatchSize)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #for epoch in range(Epochs):
        epoch = 0
        while True:
            TrainAcc = 0.0
            TrainLoss = 0.0
            for i in range(trainbatch):
                x,y = nextbatch(x_train,y_train,BatchSize,i,shuffle=True)
                if epoch>=1000:
                    _,summary,acc,loss = sess.run([apply_grads,merged_summary_op,accuracy,loss_func],feed_dict={
                    feature:x,
                    rawlabel:y
                })
                else:
                    _, summary, acc, loss = sess.run([jumper, merged_summary_op, accuracy, loss_func], feed_dict={
                        feature: x,
                        rawlabel: y
                    })
                SummaryWriter.add_summary(summary,epoch * trainbatch + i)
                TrainAcc+=acc
                TrainLoss +=loss



            valbatch = int(len(x_val)/BatchSize)

            ValAcc = 0.0
            ValLoss = 0.0
            for i in range(valbatch):
                x,y = nextbatch(x_val,y_val,BatchSize,i,shuffle=False)
                acc,loss = sess.run([accuracy,loss_func],feed_dict={
                    feature:x,
                    rawlabel:y
                })
                ValAcc+=acc
                ValLoss+=loss

            TrainAcc /= trainbatch
            TrainLoss /= trainbatch
            ValAcc/=valbatch
            ValLoss/=valbatch


            TrainAccSum = tf.Summary()
            TrainLossSum = tf.Summary()
            ValAccSum = tf.Summary()
            ValLossSum = tf.Summary()

            TrainAccSum.value.add(tag='log-by-epoch/Accuracy/Train', simple_value=TrainAcc)
            TrainLossSum.value.add(tag='log-by-epoch/Loss/Train', simple_value=TrainLoss)
            ValAccSum.value.add(tag='log-by-epoch/Accuracy/Validation',simple_value=ValAcc)
            ValLossSum.value.add(tag='log-by-epoch/Loss/Validation',simple_value=ValLoss)

            SummaryWriter.add_summary(TrainAccSum, epoch)
            SummaryWriter.add_summary(ValAccSum,epoch)
            SummaryWriter.add_summary(TrainLossSum, epoch)
            SummaryWriter.add_summary(ValLossSum,epoch)

            print('epoch %d finished at %s with \n    val-Accuracy: %.4f\n    val-Loss: %.4f' % (epoch, str(datetime.now())[11:19], ValAcc,ValLoss))
            epoch+=1


def nextbatch(x, y, BatchSize, batch_num, shuffle=False):
    if shuffle:
        extract = np.random.randint(0, len(y), BatchSize)
        rx = np.zeros((BatchSize, len(x[0])), dtype=np.float32)
        ry = np.zeros((BatchSize,), dtype=np.int32)
        for i in range(BatchSize):
            rx[i] = x[extract[i]]
            ry[i] = y[extract[i]]
            return rx, ry
    else:
        return x[BatchSize * batch_num:BatchSize * (batch_num + 1)], y[
                                                                     BatchSize * batch_num:BatchSize * (batch_num + 1)]

def baseline():
    import xgboost as xgb
    x_train,x_val,y_train,y_val = load()
    dtrain = xgb.DMatrix(x_train,y_train)
    dval = xgb.DMatrix(x_val,y_val)

    param = {'max_depth':7, 'eta':0.01, 'silent':1, 'objective':'binary:logistic'
             #,'updater':'grow_gpu',
             ,'eval_metric': ['auc','error','error@0.6']
             }

    watchlist = [(dtrain,'train'),(dval,'val')]

    num_round = 2000

    bst = xgb.train(param,dtrain,num_round,watchlist)


if __name__ == '__main__':
    #rnnfromdb()
    #fromdb()
    tftrain()
    #baseline()
    rnntest = True


