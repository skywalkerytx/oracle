from pyut import poolconn
from sklearn import preprocessing
import numpy as np

'''
thoughts:
scale for each code?
scale once for all?
min-max or gauss?
or follow the dist and write my own?
'''

con, cur = poolconn()
con.autocommit = False
cur.execute('delete from kdj')
cur.execute('select ARRAY[k,d,j] from raw order by code,date asc')
kdjs = np.asarray(list(map(lambda x: x[0], cur.fetchall())))
cur.execute(
    "select code,date,case when kdjcross='金叉' then 1 when kdjcross = '死叉' then 2 else 0 end from raw order by code,date asc")
idx = cur.fetchall()
MMS = preprocessing.MinMaxScaler()
SS = preprocessing.StandardScaler()
MMS.fit(kdjs)
SS.fit(kdjs)
kdjs = MMS.transform(kdjs)

kdj = np.zeros(shape=(len(kdjs), 15))

for i in range(len(kdjs) - 1, 4, -1):
    kdj[i] = np.concatenate((kdjs[i - 4], kdjs[i - 3], kdjs[i - 2], kdjs[i - 1], kdjs[i]), axis=0)

for i in range(5, len(kdjs)):
    code = idx[i][0]
    date = idx[i][1]
    cross = idx[i][2]
    cur.execute('insert into kdj(code,date,kdj,label) values(%s,%s,%s,%s)', (code, date, list(kdj[i]), cross))

con.commit()
con.close()
