import psycopg2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from psycopg2.pool import ThreadedConnectionPool

pool = ThreadedConnectionPool(minconn=1,maxconn=256, database='nova', user='nova', password='emeth')


def poolconn():
    conn = psycopg2.connect(database='nova', user='nova', password='emeth')
    cur = conn.cursor()
    return conn, cur

def poolconn():
    conn = pool.getconn()
    cur = conn.cursor()
    return conn,cur

def getcode():
    conn, cur = poolconn()
    cur.execute('''
SELECT
  code
FROM (
   SELECT
    code,
    count(1) as cc
   FROM
    label
   GROUP BY
    code
   ) AS countbycode
WHERE
    countbycode.cc >0;
    ''')
    _codes = cur.fetchall()
    cur.close()
    conn.close()
    return list(map(lambda x: x[0], _codes))


def getdate():
    conn, cur = poolconn()
    cur.execute('select distinct date from vector order by date asc')
    _dates = cur.fetchall()
    cur.close()
    conn.close()
    return list(map(lambda x: x[0], _dates))


codes = getcode()

dates = getdate()


def sourcefromdb(datasize = None,testsize=0.2, random_state=42,table = 'vector'):
    vectors, labels = data(datasize,table=table)
    #tvector, evector, tlabel, elabel = train_test_split(vectors, labels, test_size=testsize, random_state=random_state)
    pos = int(len(vectors)*(1-testsize))
    tvector = vectors[:pos]
    evector = vectors[pos:]
    tlabel = labels[:pos]
    elabel = labels[pos:]
    tm = xgb.DMatrix(tvector, label=tlabel)
    em = xgb.DMatrix(evector, label=elabel)
    tm.save_binary(trainingdatalocation)
    em.save_binary(testingdatalocation)
    return tm, em

trainingdatalocation = 'data/xgb/trainingdata.matrix'

testingdatalocation = 'data/xgb/testingdata.matrix'


def sourcefromfile():
    tm = xgb.DMatrix(trainingdatalocation)
    em = xgb.DMatrix(testingdatalocation)
    savetofile(tm,em)
    return tm, em


def savetofile(tm, em):
    tm.save_binary(trainingdatalocation)
    em.save_binary(testingdatalocation)


def data(datasize=None, labelnum=1,table = 'vector'):
    all = '''
    SELECT
        %s.vector,
        label.vector[%d]
    FROM
        %s
    INNER JOIN
        label
    ON
        %s.code = label.code
        AND
        %s.date = label.date
    WHERE
	%s.date < '2017-01-09'
    ORDER BY
        %s.date,%s.code ASC
    '''%(table,labelnum,table,table,table,table,table,table)
    limited = all + '\n LIMIT %s'
    con, cur = poolconn()
    if datasize is not None:
        cur.execute(limited, (datasize,))
    else:
        cur.execute(all)
    res = cur.fetchall()
    dtrain = np.asarray(list(map(lambda x: np.asarray(x[0]), res)))
    dlabel = np.asarray(list(map(lambda x: x[1], res)))
    cur.close()
    pool.putconn(con)
    return dtrain, dlabel


def getdatebycode(code, col='vector'):
    con, cur = poolconn()
    # could use parameter string but i don't care
    cur.execute('''
    SELECT
        date
    FROM ''' + col + '''

    WHERE
        code = %s
    order by date asc
    ''', (code,))
    dates = cur.fetchall()
    cur.close()
    pool.putconn(con)
    return map(lambda x: x[0], dates)


def vecbycodedate(code, date):
    con, cur = poolconn()
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
    pool.putconn(con)
    return dtrain

def labeledvector(code):
    con,cur = poolconn()
    cur.execute('''
            SELECT
                vector.vector
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
            order by vector.date asc
        ''', (code, ))
    res = np.asarray(list(map(lambda x:x[0],cur.fetchall())))
    cur.close()
    pool.putconn(con)
    return res

def unlabeledopmx(code):
    con, cur = poolconn()
    cur.execute('''
            SELECT
                vector.vector[1:2]
            FROM
                vector
            WHERE
                code = %s
            order by date asc
            ''', (code,))
    res = np.asarray(list(map(lambda x: x[0], cur.fetchall())))
    cur.close()
    pool.putconn(con)
    return res

def labelbycode(code):
    con, cur = poolconn()
    cur.execute('''
    SELECT
        vector[1]
    FROM
        label
    WHERE
        code = %s
    ORDER BY
        date asc
    ''', (code, ))
    res = cur.fetchall()
    label = list(map(lambda x:x[0],res))
    cur.close()
    pool.putconn(con)
    return label

