import xgboost as xgb
import pyut
from multiprocessing.pool import ThreadPool,Pool
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def param():
    clfparam = dict()
    clfparam['eta'] = 0.3
    clfparam['objective'] = 'binary:logistic'
    #clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 0
    clfparam['max_depth'] = 128
    #clfparam['subsample'] = 0.5

    clfparam['eval_metric'] = ['auc','error']
    #clfparam['max_delta_step'] = 1
    clfparam['scale_pos_weight'] = 29228.0/199979
    #clfparam['updater'] = 'grow_gpu'
    return clfparam

#xgb.Booster(model_file='xgb.model')

def xgbst(parameters = param()):
    tm, em = pyut.sourcefromfile()
    #tm,em = pyut.sourcefromdb(table = 'dvector')
    print("training set size: %d"%(tm.num_row()+em.num_row()))
    #watchlist = [(em, 'eval'), (tm, 'train')]
    num_round = 50
    bst = xgb.train(parameters, tm, num_round)#, watchlist)
    preds = bst.predict(em)
    elabel = em.get_label()
    positive = 0
    detected = 0
    falsealarm = 0
    print('total ' + str(len(preds)))
    count = 0
    for i in range(0, len(preds)):
        if (preds[i]>0.5) != (elabel[i]>0.5):
            count+=1
    print('error:%f'%(count/len(preds)))
    bst.save_model('data/xgb/bst.model')
    return bst

bst =xgb.Booster(model_file='data/xgb/bst.model')
#bst = xgbst()

from datetime import datetime

import numpy as np

def hista(data):

    n, bins,patches = plt.hist(data,50,normed = 0,facecolor = 'green',alpha = 0.75)
    print(n)
    print(bins)
    print(patches)
    plt.xlabel('preds')
    plt.ylabel('freq')
    plt.show()

def count(args):
    label,preds,threshold = args
    tp = ThreadPool(8)
    preds = tp.map(lambda x:x>threshold,preds)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0,len(label)):
        if label[i] == preds[i]:
            if(label[i]):
                tp+=1
            else:
                tn +=1
        else:
            if(label[i]):
                fn +=1
            else:
                fp += 1
    TPR = tp/(tp+fn)
    FPR = fp/(fp+tn)
    return (FPR,TPR,threshold)

def rocofpreds(label,preds):
    points = []
    tp = Pool(8)
    label = list(map(lambda x:x>0.5,label))
    args = list(map(lambda threshold:(label,preds,threshold),list(np.arange(0.001,1,0.001))))
    points = tp.map(count, args)
    #print(points)
    points.sort()
    x= list(map(lambda x:x[0],points))
    y= list(map(lambda x:x[1],points))
    mindist = 20
    minpt = (0,0,0)
    for point in points:
        dist = (point[0])**2 + (point[1]-1)**2
        if dist < mindist:
            minpt = point
            mindist = dist
    print(minpt)
    plt.plot(x,y,'k',np.arange(0,1,0.001),np.arange(0,1,0.001)[::-1],'b')
    plt.show()
    return minpt

def simulation():
    #vector,label = pyut.data(table='dvector')
    #print("total size %d "%len(vector))
    #xgb.DMatrix(vector,label=label).save_binary('data/xgb/validation.matrix')
    tm, em = pyut.sourcefromfile()
    #tm,em = pyut.sourcefromdb()
    #preds = bst.predict(xgb.DMatrix(vector,label=label))
    preds = bst.predict(em)
    label = em.get_label()
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    tp = []
    fp = []
    for i in range(0,len(preds)):
        if preds[i]>0.5:
            if label[i]>0.5:
                TP+=1
                tp.append(preds[i])
            else:
                FP+=1
                fp.append(preds[i])
        else:
            if label[i]<0.5:
                TN+=1
            else:
                FN+=1
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    total = len(preds)
    print('ROC point: (',FPR,TPR,')')
    print('CRITICAL ERROR: %f'%(FP/(FP+TP)))
    print('NON-CRITICAL ERROR : %f'%(FN/(TN+FN)))
    print('TP %f FP %f TN %f FN %f'%(TP/total,FP/total,TN/total,FN/total))
    pp = len(list(filter(lambda x:x>0.5,preds)))
    lp = len(list(filter(lambda x:x>0.5,label)))
    ll = len(label)
    pn = len(preds)
    print('label positive',lp,'preds positive',pp,'label neg',ll-lp,'preds neg',pn-pp)
    #hista(tp)
    #hista(fp)
    return rocofpreds(label,preds)

def predicttoday(threshold):
    con,cur = pyut.poolconn()
    cur.execute('select distinct date from raw order by date desc limit 1')
    date = cur.fetchone()[0]
    date = '2017-01-09'
    cur.execute('select code,vector,date from dvector where date = %s',(date,))
    topredict = cur.fetchall()
    tobuy = []
    for row in topredict:
        #print(row[0],bst.predict(xgb.DMatrix(np.asarray([row[1]]))))
        pred = bst.predict(xgb.DMatrix(np.asarray([row[1]])))[0]
        if pred>0.5:#threshold[2] :
            tobuy.append((pred,row[0],row[2]))

    print('total %d'%len(tobuy))

    print('sh:')
    sh =list(filter(lambda x:'sh' in x[1],tobuy))
    sh.sort(reverse=True)
    for i in range(0,15):
        print(sh[i][1],sh[i][0],sh[i][2])
    
    print('\nsz:')
    sz =list(filter(lambda x:'sh' not in x[1],tobuy))
    sz.sort(reverse=True)
    for i in range(0,15):
        print(sz[i][1],sz[i][0],sz[i][2])
    print('\n')
    print(list(map(lambda x:sh[x][1],range(0,15))) + list(map(lambda x:sz[x][1],range(0,15))))


if __name__ == '__main__':
    pass
    #pyut.getbycode('sh603166')
    #bst = xgbst()
    bst.save_model('xgb.model')
    # load model and data in
    predicttoday(simulation())
    #print(pyut.labelbycodedate('sh600027','2016-11-11'))
    #oneround(code)


