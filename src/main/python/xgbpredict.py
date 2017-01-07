import xgboost as xgb

import pyut

if __name__ == '__main__':
    (con, cur) = pyut.newconn()
    cur.execute('select code,date,vector from vector limit 10')
    res = cur.fetchall()
    map(lambda x:print(x),res)
