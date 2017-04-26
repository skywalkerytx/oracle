from pydoc import help
import psycopg2
from scipy.stats.stats import pearsonr
import numpy as np




con = psycopg2.connect(database = 'nova',user = 'nova',password = 'emeth')



help(pearsonr)

cur = con.cursor()

cur.execute('''
SELECT
    k,d,j,
    case when kdjcross='金叉' then 1 
    when kdjcross='死叉' then 0
    else 0 end as kdj,
    vector[1]
FROM
    raw
INNER JOIN
    label
ON 
    raw.code = label.code
AND
    raw.date = label.date
''')

res = cur.fetchall()

matrix = [[],[],[],[],[]]
for i in range(5):
    vector = [x[i] for x in res]
    matrix[i] = vector

matrix = np.asarray(matrix)

name = ['k','d','j','kdjcross','label']

for i in range(4):
    r,p = pearsonr(matrix[i],matrix[-1])
    print('between %s and %s, the pearson R is %f, and p is %f'%(name[i],name[-1],r,p))

