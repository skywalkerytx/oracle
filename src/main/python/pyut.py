import psycopg2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from psycopg2.pool import ThreadedConnectionPool

SqlPool = ThreadedConnectionPool(minconn=1, maxconn=256, database='nova', user='nova', password='emeth')


def conn():
    conn = psycopg2.connect(database='nova', user='nova', password='emeth')
    cur = conn.cursor()
    return conn, cur

def poolconn():
    conn = SqlPool.getconn()
    cur = conn.cursor()
    return conn,cur


def putcon(con):
    SqlPool.putconn(con)

def getcode():
    conn, cur = poolconn()
    cur.execute('select distinct code from raw')
    res = map(lambda x: x[0], cur.fetchall())
    return list(res)


def getdate(code):
    conn, cur = poolconn()
    cur.execute('select distinct date from raw where code = %s order by date asc', (code,))
    res = map(lambda x: x[0], cur.fetchall())
    return list(res)


def dictoffeature(code):
    conn, cur = poolconn()
    cur.execute('select date,ARRAY[k,d,j] from raw where code = %s', (code,))
    res = cur.fetchall()
    return dict(res)
