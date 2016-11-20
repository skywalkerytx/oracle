import mysql.connector as maria
import os

from multiprocessing.pool import Pool

stock_header = '股票代码,股票名称,交易日期,新浪行业,新浪概念,新浪地域,开盘价,最高价,最低价,收盘价,后复权价,前复权价,涨跌幅,成交量,成交额,换手率,流通市值,总市值,是否涨停,是否跌停,市盈率TTM,市销率TTM,市现率TTM,市净率,MA_5,MA_10,MA_20,MA_30,MA_60,MA金叉死叉,MACD_DIF,MACD_DEA,MACD_MACD,MACD_金叉死叉,KDJ_K,KDJ_D,KDJ_J,KDJ_金叉死叉,布林线中轨,布林线上轨,布林线下轨,psy,psyma,rsi1,rsi2,rsi3,振幅,量比\n'
new_stock_header = "code,name,date,industry,concept,area,op,mx,mn,clse,aft,bfe,amp,vol,market,market_exchange,on_board,total,ZT,DT,shiyinlv,shixiaolv,shixianlv,shijinglv,ma5,ma10,ma20,ma30,ma60,macross,macddif,macddea,macdmacd,macdcross,k,d,j,kdjcross,berlinmid,berlinup,berlindown,psy,psyma,rsi1,rsi2,rsi3,zhenfu,volratio"

index_header = 'index_code,date,open,close,low,high,volume,money,change\n'
new_index_header = "index_code,index_date,open,close,low,high,volume,money,delta"

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

def AddStock(stock):
    pass

def AddIndex(index):
    pass

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