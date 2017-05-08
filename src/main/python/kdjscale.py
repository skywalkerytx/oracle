import psycopg2

con = psycopg2.connect(database='nova', user='nova', password='emeth')

cur = con.cursor()

codes27 = ['sz002054', 'sz002509', 'sz002304', 'sz300363', 'sz300172', 'sz300055', 'sh600195', 'sz002643']  # 27
codes26 = ['sh600742', 'sh603369', 'sh600388']  # 26
codes28 = ['sh600477', 'sh600211', 'sz300122']
codes = codes27
# codes.extend(codes27)
# codes.extend(codes28)
date = '2017-04-27'

for code in codes:
    cur.execute('SELECT date FROM raw WHERE date >%s ORDER BY date ASC LIMIT 1', (date,))
    nextdate = cur.fetchone()[0]
    nextdate = date
    cur.execute('SELECT op FROM raw WHERE code = %s AND date=%s', (code, nextdate))
    op = cur.fetchone()[0]
    cur.execute('SELECT date,mx FROM raw WHERE code = %s AND date > %s AND mx >= %s ORDER BY date ASC',
                (code, nextdate, op * 1.03))
    cc = cur.fetchone()

    if cc is not None:
        s = code + ',' + date + ',' + nextdate + ',' + str(op) + ',' + str(cc[0]) + ',' + str(cc[1])
    else:
        s = code + ',' + date + ',' + nextdate + ',' + str(op) + ',' + 'NOT SELLED YET, NAN'

    print(s)
