import pymysql
from mysql.connector.pooling import MySQLConnectionPool as mariapool
import mysql.connector as maria

#savior of day: echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse

def newconn():
    conn = maria.connect(user='nova', db = 'nova',charset = 'utf8mb4',host = 'localhost')
    cur = conn.cursor()
    return (conn,cur)


def newpool(size = 8):
    return mariapool(pool_size=size,user='nova',database='nova', charset = 'utf8mb4',host = 'localhost')