import pandas as pd
import h5py
import numpy as np
import mysql.connector
from urllib.parse import urlparse

dat=pd.read_csv("../data/TIC3.list")
rad=[]
mass=[]
#mysql -h 133.11.231.118 -u {user} -p TESS
url = urlparse('mysql://fisher:atlantic@133.11.231.118:3306/TESS')
conn = mysql.connector.connect(
        host = url.hostname or '133.11.231.118',
        port = url.port or 3306,
        user = url.username or 'fisher',
        password = url.password or 'atlantic',
        database = url.path[1:],
)
cur = conn.cursor()
j=0
nlen=len(dat["FILE"])
for fn in dat["FILE"]:
    j=j+1
    if(np.mod(j,100000)==0):
        print(j,"/",nlen)
    tic=fn.split("_")[1]
    try:
        cur.execute('SELECT rad,mass FROM CTLv7 where ID='+str(tic))
        out=cur.fetchall()[0]
        out=np.array(out) #rad, mass
        rad.append(out[0])
        mass.append(out[1]) 
    except:
        rad.append(-1.0)
        mass.append(-1.0)
rad=np.array(rad)
mass=np.array(mass)
np.savez("TIC3.list.npz",dat["FILE"],rad,mass)
