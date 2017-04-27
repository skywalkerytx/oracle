from pydoc import help
import psycopg2
from scipy.stats.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt




con = psycopg2.connect(database = 'nova',user = 'nova',password = 'emeth')

cur = con.cursor()

extra = "and k>%s and raw.date >='2016-03-01'"
X = np.arange(10,80,1)
sr = np.zeros(len(X))
recall = np.zeros(len(X))
count = 0
for lower_bound in X:
    lb = int(lower_bound)
    cur.execute('''
    SELECT                              
      count(1)
    FROM
      raw
    INNER JOIN label
      ON raw.code = label.code AND raw.date = label.date
    WHERE
      raw.kdjcross='金叉' and raw.macdcross='金叉' and label.vector[1]=0 '''+extra,(lb,))
    fail = cur.fetchone()[0]
    cur.execute('''
    SELECT                              
      count(1)
    FROM
      raw
    INNER JOIN label
      ON raw.code = label.code AND raw.date = label.date
    WHERE
      raw.kdjcross='金叉' and raw.macdcross='金叉' and label.vector[1]=1 '''+extra,(lb,))
    success = cur.fetchone()[0]

    total = fail+success
    cr = 1.0*success/total
    willpick = total/384222.0*2575
    rightpick = success/384222.0*2575
    print('%d %.4f will pick:%.4f right amount: %.4f'%(lb,cr,willpick,rightpick))
    sr[count] = cr
    recall[count] = rightpick
    count = count +1
plt.figure(1)
plt.subplot(211)
plt.plot(X,sr,'r')
plt.subplot(212)
plt.plot(X,recall,'b')
plt.show()

cur.execute("""
select
code,k 
from 
raw
WHERE
1=1
and raw.kdjcross='金叉'
and raw.macdcross='金叉'
and date > '2017-04-25 ' 
and date < '2017-04-27 '
order by k desc
""")
result = cur.fetchall()
#print(result)
from datetime import datetime
filename = str(datetime.now())[0:10]+'.csv'
f = open(filename,'w')
f.write('code,prob\n')
for line in result:
    code = line[0]
    k = line[1]
    pos = len(X)-1
    for j in range(1,len(X)):
        if X[j-1]<=k and X[j]>=k:
            pos = j-1
            break
    s = code+',>'+str(sr[pos])[0:6]+'\n'
    f.write(s)
    print(s,k)

f.close()
