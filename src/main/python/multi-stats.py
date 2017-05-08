import psycopg2
from matplotlib import pyplot as plt

con = psycopg2.connect(database='nova', user='nova', password='emeth')
cur = con.cursor()


def getamp(code, date):  # buy at cross day
    cur.execute('SELECT op FROM raw WHERE code = %s AND date = %s', (code, date))
    op = cur.fetchone()[0]
    cur.execute('SELECT mx FROM raw WHERE code = %s AND date > %s ORDER BY date ASC LIMIT 1', (code, date))
    try:
        mx = cur.fetchone()[0]
        # print(date, mx[1])
    except TypeError:
        return None

    return 1.0 * mx / op


def getlateramp(code, date):
    cur.execute('SELECT op,mx FROM raw WHERE code = %s AND date > %s ORDER BY date ASC LIMIT 2', (code, date))
    try:
        op, mx = cur.fetchall()
    except ValueError:
        return None
    op = op[0]
    mx = mx[1]
    return 1.0 * mx / op


cur.execute("SELECT code,date FROM raw WHERE kdjcross='金叉' AND macdcross = '金叉'")

res = cur.fetchall()

amps = []

for code, date in res:
    amp = getlateramp(code, date)
    if amp is None:
        continue
    amps.append(amp)

plt.figure(1)
x, bins, patches = plt.hist(amps, bins=50)
plt.figure(2)
total = len(amps)
ss = [sum(x[0:i]) * 1.0 / total for i in range(len(x))]
plt.plot(bins[0:50], ss)

plt.show()
