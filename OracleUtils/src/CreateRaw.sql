CREATE TABLE IF NOT EXISTS Raw (
code char(9) NOT NULL,
name varchar(12) NOT NULL,
date char(11) NOT NULL,
industry text NOT NULL,
concept text NOT NULL,
area text NOT NULL,
op FLOAT NOT NULL,
mx FLOAT NOT NULL,
mn FLOAT NOT NULL,
clse FLOAT NOT NULL,
aft FLOAT NOT NULL,
bfe FLOAT NOT NULL,
amp FLOAT NOT NULL,
vol FLOAT NOT NULL,
market FLOAT NOT NULL,
market_exchange FLOAT NOT NULL,
on_board FLOAT NOT NULL,
total FLOAT NOT NULL,
ZT FLOAT NOT NULL,
DT FLOAT NOT NULL,
shiyinlv FLOAT NOT NULL,
shixiaolv FLOAT NOT NULL,
shixianlv FLOAT NOT NULL,
shijinglv FLOAT NOT NULL,
ma5 FLOAT NOT NULL,
ma10 FLOAT NOT NULL,
ma20 FLOAT NOT NULL,
ma30 FLOAT NOT NULL,
ma60 FLOAT NOT NULL,
macross text NOT NULL,
macddif FLOAT NOT NULL,
macddea FLOAT NOT NULL,
macdmacd FLOAT NOT NULL,
macdcross text NOT NULL,
k FLOAT NOT NULL,
d FLOAT NOT NULL,
j FLOAT NOT NULL,
kdjcross text NOT NULL,
berlinmid FLOAT NOT NULL,
berlinup FLOAT NOT NULL,
berlindown FLOAT NOT NULL,
psy FLOAT NOT NULL,
psyma FLOAT NOT NULL,
rsi1 FLOAT NOT NULL,
rsi2 FLOAT NOT NULL,
rsi3 FLOAT NOT NULL,
zhenfu FLOAT NOT NULL,
volratio FLOAT NOT NULL,
PRIMARY KEY (code,date)
);
CREATE TABLE IF NOT EXISTS RawIndex(
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


CREATE TABLE IF NOT EXISTS Mapping(
str varchar(64),
cat varchar(12),
gid serial,
PRIMARY KEY(cat,str)
);

CREATE TABLE IF NOT EXISTS RawConcept(
concept varchar(20),
amount int,
uppercent float,
downpercent float,
drawpercent float,
amp float,
wamp float,
aprofit float,
date varchar(11),
PRIMARY KEY(concept,date)
);

CREATE INDEX IF NOT EXISTS RawConcept_date_idx ON RawConcept (date);

CREATE INDEX IF NOT EXISTS RawIndex_index_date_idx ON RawIndex (index_date);

CREATE TABLE IF NOT EXISTS Vector(
code char(9),
date char(11),
vector float[],
PRIMARY KEY (code,date)
);

CREATE INDEX IF NOT EXISTS Vector_code_idx ON Vector(code);

CREATE INDEX IF NOT EXISTS Vector_date_idx ON Vector(date);

CREATE TABLE IF NOT EXISTS Label(
code char(9),
date char(11),
vector float[],
PRIMARY KEY (code,date)
);

CREATE INDEX IF NOT EXISTS Label_code_idx ON Label(code);

CREATE INDEX IF NOT EXISTS Label_date_idx ON Label(date);