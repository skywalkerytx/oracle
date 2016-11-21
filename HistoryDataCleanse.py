import mysql.connector as maria
import os
import re

from multiprocessing.pool import Pool

CreateTableSQL = '''
CREATE OR REPLACE TABLE Raw (
code char(9) NOT NULL,
name varchar(12) NOT NULL,
date char(11) NOT NULL,
industry text NOT NULL,
concept text NOT NULL,
area text NOT NULL,
op double precision NOT NULL,
mx double precision NOT NULL,
mn double precision NOT NULL,
clse double precision NOT NULL,
aft double precision NOT NULL,
bfe double precision NOT NULL,
amp double precision NOT NULL,
vol double precision NOT NULL,
market double precision NOT NULL,
market_exchange double precision NOT NULL,
on_board double precision NOT NULL,
total double precision NOT NULL,
ZT double precision NOT NULL,
DT double precision NOT NULL,
shiyinlv double precision NOT NULL,
shixiaolv double precision NOT NULL,
shixianlv double precision NOT NULL,
shijinglv double precision NOT NULL,
ma5 double precision NOT NULL,
ma10 double precision NOT NULL,
ma20 double precision NOT NULL,
ma30 double precision NOT NULL,
ma60 double precision NOT NULL,
macross text NOT NULL,
macddif double precision NOT NULL,
macddea double precision NOT NULL,
macdmacd double precision NOT NULL,
macdcross text NOT NULL,
k double precision NOT NULL,
d double precision NOT NULL,
j double precision NOT NULL,
kdjcross text NOT NULL,
berlinmid double precision NOT NULL,
berlinup double precision NOT NULL,
berlindown double precision NOT NULL,
psy double precision NOT NULL,
psyma double precision NOT NULL,
rsi1 double precision NOT NULL,
rsi2 double precision NOT NULL,
rsi3 double precision NOT NULL,
zhenfu double precision NOT NULL,
volratio double precision NOT NULL,
PRIMARY KEY (code,date)
);
CREATE OR REPLACE TABLE RawIndex(
index_code char(9) NOT NULL,
index_date char(11) NOT NULL,
open float NOT NULL,
close float NOT NULL,
low float NOT NULL,
high float NOT NULL,
volume float NOT NULL,
money float NOT NULL,
delta float NOT NULL,
PRIMARY KEY (index_code,index_date)
);
'''

conn = maria.connect(user = 'nova', database = 'nova')
cur = conn.cursor()


def CreateTable():
    cur_iter = cur.execute(CreateTableSQL, multi=True)
    for i in cur_iter:
        pass  # Have to,otherwise won't work anyway

    conn.commit()  # autocommit = False by default
    cur.execute('select * from Raw limit 1')


def AddStock(stock):
    if stock[0]!='s':
        return
    pconn = maria.connect(user='nova', database='nova')
    pcur = pconn.cursor()
    if stock[1] == 'h':
        stock = 'data/history/overview-data-sh/'+stock
    else:
        stock = 'data/history/overview-data-sz/'+stock
    FILE = open(stock,'r',encoding='gbk')
    #ensure no ridiculous data get into database
    lines = filter(lambda line:re.match('''^[sh0-9]+,[^,]*,[\-0-9]{10,10},[^,]*,[^,]*,[^,]*,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[^,]*,[.0-9]+,[.0-9]+,[.0-9]+,[^,]*,[.0-9]+,[.0-9]+,[.0-9]+,[^,]*,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+$''',line)!=None,FILE.readlines())
    pcur.executemany('''
    INSERT INTO Raw(
    code,
    name,
    date,
    industry,
    concept,
    area,
    op,
    mx,
    mn,
    clse,
    aft,
    bfe,
    amp,
    vol,
    market,
    market_exchange,
    on_board,
    total,
    ZT,
    DT,
    shiyinlv,
    shixiaolv,
    shixianlv,
    shijinglv,
    ma5,
    ma10,
    ma20,
    ma30,
    ma60,
    macross,
    macddif,
    macddea,
    macdmacd,
    macdcross,
    k,
    d,
    j,
    kdjcross,
    berlinmid,
    berlinup,
    berlindown,
    psy,
    psyma,
    rsi1,
    rsi2,
    rsi3,
    zhenfu,
    volratio)
    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ''',list(map(lambda x:x.split(','),lines)))

    pconn.commit()
    pconn.close()
    FILE.close()



def AddIndex(index):
    if index[0] != 's':
        return
    pconn = maria.connect(user='nova',database = 'nova')
    pcur = pconn.cursor()
    index = 'data/history/overview-data-sh/index/'+index
    FILE = open(index,'r',encoding='gbk')
    lines = filter(lambda line: re.match('''^[sh0-9]+,[\-0-9]{10,10},[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,[.0-9]+,$''',line)!=None,FILE.readlines())
    pcur.executemany('''
    INSERT INTO RawIndex(
    index_code,index_date,open,close,low,high,volume,money,delta) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ''',list(map(lambda x:x.split(','),lines)))

    pconn.commit()
    pconn.close()
    FILE.close()

def main():
    CreateTable()
    stock = []
    index = []
    for [a,b,c] in os.walk('data/history'):
        if a == 'data/history/overview-data-sh':
            stock += c
        if a == 'data/history/overview-data-sh/index':
            index = c
        if a == 'data/history/overview-data-sz':
            stock += c
    tp = Pool(8)
    tp.map(AddStock,stock)
    tp.map(AddIndex,index)
    return 0

if __name__=='__main__':
    exit(main())