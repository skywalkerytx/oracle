import mysql.connector as maria
from multiprocessing.pool import Pool
from utils.globalvar import cats

TableSQL = '''
CREATE OR REPLACE TABLE Mapping(
str varchar(64),
cat varchar(12),
id int,
PRIMARY KEY(cat,str)
);
'''



def CatMapping(cat):
    print(cat)
    pconn = maria.connect(user='nova', database='nova')
    pcur = pconn.cursor()
    pcur.execute('select distinct %s from Raw'%cat)
    lines = list(map(lambda line:line[0].decode('UTF-8').split('；'),pcur))
    mappings = dict()
    for line in lines:
        for word in line:
            if word not in mappings:
                mappings[word] = len(mappings)
    for word in mappings.keys():
        pcur.execute('INSERT INTO Mapping(str,cat,id) VALUES(%s,%s,%s)',(word,cat,mappings[word]))
    pconn.commit()
    pconn.close()


def main():
    conn = maria.connect(user='nova', database='nova')
    cur = conn.cursor()
    cur.execute(TableSQL)
    conn.commit()
    tp = Pool(6)
    tp.map(CatMapping, cats)
    return 0

if __name__ == '__main__':
    exit(main())