import mysql.connector as maria
from utils.globalvar import cats
from functools import reduce
from utils.newconn import newpool
from multiprocessing.pool import Pool,ThreadPool


import numpy as np

catslen= dict()

catsmap = dict()

codes = []
dates = []

SqlPool = newpool(2)

ProcessSqlPool = None


def catmapping(words,cat):
    mapping = np.zeros(catslen[cat])
    pos = map(lambda word:catsmap[cat][word],words)
    for p in pos:
        mapping[p] = 1
    return mapping

def getlens():
    conn = SqlPool.get_connection()
    cur = conn.cursor()
    cur.execute('select cat,count(1) from Mapping group by cat')
    res =  dict(map(lambda x:(x[0].decode('utf8'),x[1]),cur))
    conn.close()
    return res

def getmapping(cat):
    conn = SqlPool.get_connection()
    cur = conn.cursor()
    cur.execute('select str,id from Mapping where cat = %s',(cat,))
    res = dict(map(lambda x:(x[0].decode('utf8'),x[1]),cur))
    conn.close()
    return res

def getmappings():
    return dict(map(lambda cat:(cat,getmapping(cat)),cats))

def mappingarray(args):
    (code, date) = args
    query = 'select ' + reduce(lambda x,y:x+','+y,cats) + ' from Raw where code = %s and date = %s'
    conn = ProcessSqlPool.get_connection()
    cur = conn.cursor()
    cur.execute(query, (code, date))
    for line in cur:
        if cur.rowcount == -1:
            return
        line = list(map(lambda x:x.decode('utf8').split('；'),line))
        array = reduce(lambda x,y:np.append(x,y),map(lambda iter:catmapping(line[iter],cats[iter]),range(0,len(cats))))
    conn.close()


def GenC(code):
    tp = ThreadPool(8)
    args = tp.map(lambda date:(code,date),dates)
    tp.map(mappingarray,args)
    #list(map(mappingarray,args))
    tp.terminate()

def ProcessInitializer():
    global ProcessSqlPool
    ProcessSqlPool = newpool(8)

def main():
    global catslen
    global catsmap
    global codes,dates
    catslen = getlens()
    catsmap = getmappings()
    # print(catsmap)
    # mappingarray('sh600000','2001-10-26')
    conn = SqlPool.get_connection()
    cur = conn.cursor()
    cur.execute('select distinct code from Raw')
    codes = list(map(lambda x: x[0].decode('utf8'), cur))
    cur.execute('select distinct date from Raw')
    dates = list(map(lambda x: x[0].decode('utf8'), cur))
    conn.close()
    pp = Pool(8,initializer=ProcessInitializer)
    pp.map(GenC, codes)

if __name__ == '__main__':
    exit(main())
