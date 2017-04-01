import xgboost as xgb
from multiprocessing.pool import ThreadPool,Pool
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pyut


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


if __name__ == '__main__':
    res = pyut.dictoffeature('sh600021')
    for line in res:
        print(line, res[line])
