from datetime import datetime

def timer(func,*args):
    a = datetime.now()
    func(*args)
    b = datetime.now()
    return ((b-a).seconds,a,b)