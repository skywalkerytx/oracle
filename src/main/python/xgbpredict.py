import xgboost as xgb
import pyut


def param():
    clfparam = dict()
    clfparam['booster'] = 'gbtree'
    clfparam['silent'] = 0
    clfparam['max_depth'] = 6
    clfparam['subsample'] = 0.5
    return clfparam

bst = xgb.Booster(model_file='xgb.model')

def xgbst():
    tm, em = pyut.sourcefromfile()
    watchlist = [(em, 'eval'), (tm, 'train')]
    num_round = 2
    bst = xgb.train(param(), tm, num_round, watchlist)
    preds = bst.predict(em)
    elabel = em.get_label()
    positive = 0
    detected = 0
    falsealarm = 0
    print('total ' + str(len(preds)))
    for i in range(0, len(preds)):
        if preds[i] > 0.5:
            detected += 1
            if elabel[i] == 1:
                positive += 1
            else:
                falsealarm += 1
    print(positive * 100.0 / detected, falsealarm * 100.0 / detected)
    return bst

from datetime import datetime

import numpy as np

def oneround(code):
    st = datetime.now()
    detected = 0
    mismatch = 0
    wronglabel = 0
    loss = 0
    wincount = 0
    dates = list(pyut.getdatebycode(code, 'vector'))
    labels = pyut.labelbycode(code)
    unlabeled = pyut.unlabeledopmx(code)
    preds =bst.predict(xgb.DMatrix(pyut.labeledvector(code)))
    for i in range(len(preds)):
        pred = preds[i]
        daftt = dates[i + 2]
        dt = dates[i + 1]
        tormorrow, dayafter = unlabeled[i]
        label = labels[i]
        detected += 1
        if pred > 0.5:
            if label != 1:
                mismatch += 1
                loss += dayafter / tormorrow
            else:
                wincount += 1
        if label == 1 and tormorrow * 1.03 < dayafter:
            wronglabel += 1
    et = datetime.now()
    print(code + ' ends. time cost:'+str((et-st).seconds))
    return detected,mismatch,wronglabel,loss,wincount


def simulation():

    from multiprocessing.pool import ThreadPool
    import multiprocessing
    tp = ThreadPool(multiprocessing.cpu_count())
    total = tp.map(oneround,pyut.codes)
    detected = sum(map(lambda x:x[0],total))
    mismatch = sum(map(lambda x:x[1],total))
    wronglabel = sum(map(lambda x:x[2],total))
    loss = sum(map(lambda x:x[3],total))
    wincount = sum(map(lambda x:x[4],total))
    print("wrong rate: %f"%(mismatch*1.0/detected))
    print("avg loss: %f"%(loss/mismatch))
    print("profit rate: %f"%((wincount*1.03-loss)/detected))
    print('wrong label %d'%wronglabel)




if __name__ == '__main__':
    #pyut.getbycode('sh603166')
    #bst = xgbst()
    #bst.save_model('xgb.model')
    # load model and data in
    print("start")
    code = 'sz002316'
    print(pyut.labelbycode(code))
    simulation()
    #print(pyut.labelbycodedate('sh600027','2016-11-11'))
    #oneround(code)


