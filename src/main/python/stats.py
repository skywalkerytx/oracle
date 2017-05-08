# coding=utf-8
import psycopg2
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

latests = []
GeneratePast = False

if GeneratePast:
    latests = ['2017-04-26', '2017-04-27', '2017-04-28']
else:
    cur.execute('SELECT DISTINCT date FROM raw ORDER BY date DESC LIMIT 1')
    latests = [cur.fetchone()[0]]

for latest in latests:
    print(latest)

    cur.execute("""
    SELECT
    code,k 
    FROM 
    raw
    WHERE
    1=1
    AND raw.kdjcross='金叉'
    AND raw.macdcross='金叉'
    AND date = %s 
    ORDER BY k DESC
    """, (latest,))
    result = cur.fetchall()
    filename = 'data/result/' + latest + '.csv'
    print('writing %s' % filename)
    f = open(filename, 'w')
    f.write('code,prob\n')
    Sorted = []
    for line in result:
        code = line[0]
        k = line[1]
        pos = len(X) - 1
        for j in range(1, len(X)):
            if X[j - 1] <= k and X[j] >= k:
                pos = j - 1
                break
        Sorted.append((sr[pos], code))

    Sorted = sorted(Sorted, reverse=True)

    for line in Sorted:
        code = line[1]
        prob = line[0]
        print(code, prob)
        if prob <= 0.52:
            continue
        s = code + ',>' + str(prob)[0:6] + '\n'
        f.write(s)
        print(s)
    f.close()
