import psycopg2
import numpy as np


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
        vector.code,vector.date asc
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
    return dtrain, dlabel
