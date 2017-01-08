import psycopg2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def newconn():
    conn = psycopg2.connect(database='nova', user='nova', password='emeth')
    cur = conn.cursor()
    return conn, cur


def getcode():
    conn, cur = newconn()
    cur.execute('select distinct code from vector order by code asc')
    _codes = cur.fetchall()
    cur.close()
    conn.close()
    return list(map(lambda x: x[0], _codes))

def getdate():
    conn, cur = newconn()
    cur.execute('select distinct date from vector order by date asc')
    _dates = cur.fetchall()
    cur.close()
    conn.close()
    return list(map(lambda x: x[0], _dates))

codes = getcode()

dates = getdate()

def sourcefromdb(testsize=0.2,random_state=42):
     vectors, labels = data(labelnum=0)
     tvector, evector, tlabel, elabel = train_test_split(vectors, labels, test_size=testsize, random_state=random_state)
     tm = xgb.DMatrix(tvector, label=tlabel)
     em = xgb.DMatrix(evector, label=elabel)
     return tm,em

trainingdatalocation = 'data/xgb/trainingdata.matrix'

testingdatalocation = 'data/xgb/testingdata.matrix'

def sourcefromfile():
    tm = xgb.DMatrix(trainingdatalocation)
    em = xgb.DMatrix(testingdatalocation)
    return tm,em

def savetofile(tm,em):
    tm.save_binary(trainingdatalocation)
    em.save_binary(testingdatalocation)

def data(datasize = None, labelnum = 0):

    all = '''
    SELECT
        vector.vector,
        label.vector
    FROM
        vector
    INNER JOIN
        label
    ON
        vector.code = label.code
        AND
        vector.date = label.date
    ORDER BY
        vector.code,vector.date ASC
    '''

    limited =all+'\n LIMIT %s'

    con, cur = newconn()

    if datasize is not None:
        cur.execute(limited, (datasize,))
    else:
        cur.execute(all)
    res = cur.fetchall()
    dtrain = np.asarray(list(map(lambda x: np.asarray(x[0]), res)))
    dlabel = np.asarray(list(map(lambda x: x[1][labelnum], res)))
    cur.close()
    con.close()
    return dtrain, dlabel


def getdatebycode(code,col = 'vector'):
    con,cur = newconn()
    #could use parameter string but i don't care
    cur.execute('''
    SELECT
        date
    FROM ''' + col + '''

    WHERE
        code = %s
    order by date asc
    ''',(code,))
    dates = cur.fetchall()
    cur.close()
    con.close()
    return map(lambda x:x[0],dates)

def vecbycodedate(code, date):
    con,cur = newconn()
    cur.execute('''
    SELECT
        vector
    FROM
        vector
    WHERE
        code = %s
        AND
        date = %s
    ''', (code, date))
    res = cur.fetchone()
    dtrain = np.asarray(res)
    cur.close()
    con.close()
    return dtrain

def labelbycodedate(code,date):
    con,cur = newconn()
    cur.execute('''
    SELECT
        vector
    FROM
        label
    WHERE
        code = %s
        AND
        date = %s
    ''', (code, date))
    res = cur.fetchone()
    label = res[0]
    cur.close()
    con.close()
    return label[0]