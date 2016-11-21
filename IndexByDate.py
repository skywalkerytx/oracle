from utils.newconn import newpool


SqlPool = None

def PoolInitializer():
    global SqlPool
    SqlPool = newpool()

def IndexByDate(date):
    con  = SqlPool.get_connection()
    cur = con.cursor()
    cur.execute('selct open,close,low,high,volume,money,delta where index_date = %s order by index_code asc',(date,))


def main():


    return 0



if __name__ == '__main__':
    exit(main())