import mysql.connector as maria
from multiprocessing.pool import Pool

TableSQL = '''
CREATE OR REPLACE TABLE Mapping(
str varchar(64),
cat varchar(12),
id int,
PRIMARY KEY(cat,str)
);
'''

cats = "industry,concept,area,macross,macdcross,kdjcross".split(',') #meow


def CatMapping(cat):
    print(cat)
    pconn = maria.connect(user='nova', database='nova')
    pcur = pconn.cursor()
    pcur.execute('select distinct %s from Raw'%cat)
    for line in pcur:
        print(line[0].decode('UTF-8')) #never needed in psql



def main():
    conn = maria.connect(user='nova', database='nova')
    cur = conn.cursor()
    cur.execute(TableSQL)
    conn.commit()
    tp = Pool(6)
    #tp.map(CatMapping,cats)
    list(map(CatMapping,cats))

    return 0

if __name__ == '__main__':
    exit(main())