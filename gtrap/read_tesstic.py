#
# This code reads LC from ffi by Tajiri system!
#
import numpy as np 
import h5py


def load_tesstic(filelist,offt="t[0]",nby=1000):
    
    inval=-1000.0 #invalid time
    nq=1 # # of bacth
    lc=[]
    tu=[]
    ntrue=[]
    t0arr=[]
    nfile=len(filelist)

    
    for k in range(0,nq):
        t, det, q, cno, ra, dec =read_tesstic(filelist)

        n=int(cno[-1]-cno[0]+1)
        while np.mod(n,nby)>0:
            n=n+1
        
        t0arr.append(t[0])
        
        lcn, tun=throw_tessintarray(n,cno,t,det,fillvalv=1.002,fillvalt=inval,offt=offt)        
        gapmask=(tun<0)
        ntrue.append(len(tun[~gapmask]))
        Hfill=np.max(lcn)
        Lfill=np.min(lcn)
        maskL=gapmask[::2]
        maskH=gapmask[1::2]   
        lcn[::2][maskL]=Lfill
        lcn[1::2][maskH]=Hfill
        tu.append(tun)
        lc.append(lcn)
    lc=np.array(lc).transpose().astype(np.float32)
    tu=np.array(tu).transpose().astype(np.float32)
    ntrue=np.array(ntrue).astype(np.uint32)

    ##original masked data
#    mask=(t==t)
#    t=t[mask]
#    det=det[mask]
    return lc,tu,n,ntrue,nq,inval


def read_tesstic(hdf):

    with h5py.File(hdf,"r") as f:
        
        time=f["LC"]["TIME"].value
        flux=f["LC"]["SAP_FLUX"].value
        q=f["LC"]["QUALITY"].value

        dta=time[1:]-time[:-1]
        dt = np.median(dta)
        cno = np.round((time-time[0])/dt).astype(np.int32)
        #error=(time - (cno*dt + time[0]) )
        #print(np.max(error))
        
        ra=f["header"]["ra"].value                        
        dec=f["header"]["dec"].value                        

        return time, flux, q, cno, ra, dec


def throw_tessintarray(n,cno,t,lc,fillvalv=-1.0,fillvalt=-5.0,offt="t[0]"):
    #dt=t[2]-t[1]
    #offt=t[1]
    offset=np.array(cno[0])
    jend=int(cno[-1]-offset+1)
    lcn=np.ones(n)*fillvalv
    if(jend > n):
        print("Set larger n than ",jend)
        sys.exit("Error")
#    else:
#        print("Filling ",jend,"-values in ",n," elements in lcn.")
    tun=np.ones(n)*fillvalt
    if offt=="t[0]":
        t0=t[0]
    else:
        t0=0.0

    for i in range(0,len(cno)):
        j=int(cno[i]-offset)
        if lc[i]==lc[i]:    
            lcn[j]=lc[i]
            tun[j]=t[i]-t0

    return lcn,tun

    
if __name__ == "__main__":
    time, flux, q, cno, ra, dec = read_tesstic("/pike/pipeline/step3/tess_1111192_8_2_4.h5")
#    lcn,tun = throw_tessintarray(n,cno,t,lc,fillvalv=-1.0,fillvalt=-5.0,offt="t[0]")
    
