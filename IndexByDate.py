from utils.newconn import newpool,newconn
from multiprocessing.pool import Pool
import numpy as np


SqlPool = None

def PoolInitializer():
    global SqlPool
    SqlPool = newpool(size=2)

def IndexByDate(date):
    con  = SqlPool.get_connection()
    cur = con.cursor()
    cur.execute('select open,close,low,high,volume,money,delta from RawIndex where index_date = %s order by index_code asc',(date,))
    res = np.array(cur.fetchall()).reshape(-1)
    con.close()
    return res

def main():
    pp = Pool(8,initializer=PoolInitializer)
    (conn,cur) = newconn()
    cur.execute('select distinct index_date from RawIndex order by index_date asc')
    dates = map(lambda x:x[0].decode('utf8'),cur.fetchall())
    cur.execute('select index_date,count(index_code) as cnt from RawIndex group by index_date order by index_date desc')
    print(cur.fetchall())
    res = pp.map(IndexByDate,dates)
    return 0



if __name__ == '__main__':
    exit(main())