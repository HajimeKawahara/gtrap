#
# This code reads LC from ffi by Tajiri system!
#
import numpy as np 
import h5py
import sys
import pandas as pd
from gtrap import asteroid_indicator as ai
from gtrap import calcbkgd  

def load_tesstic(filelist,n,offt="t[0]",nby=1000,good_quality=True):
    
    inval=-1000.0 #invalid time
    nq=len(filelist) # # of bacth
    lc=[]
    tu=[]
    asind=[]
    bkgu=[]
    infogap=[]
    ntrue=[]
    tu0=[]
    ticarr=[]
    sectorarr=[]
    cameraarr=[]
    CCDarr=[]
    while np.mod(n,nby)>0:
        n=n+1

    for k in range(0,nq):
        t, det, q, cno, ra, dec, tic, sector, camera, CCD, cnts, bkg =read_tesstic(filelist[k])

        ndmax = ai.comp_ndmax(cnts)
        
        if good_quality:
            #masking quality flaged bins
            mask=(q==0) + (q==2)
#            mask=(q==0)
            t=t[mask]
            det=det[mask]
            cno=cno[mask]
            ndmax=ndmax[mask]
            
        lcn, tun, ndmaxn, bkgn, t0=throw_tessintarray(n,cno,ndmax,bkg,t,det,fillvalv=1.002,fillvalt=inval,offt=offt)        

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
        bkgu.append(bkgn)
        asind.append(ndmaxn)
#        infogap.append(igap)
        
        tu0.append(t0)
        ticarr.append(tic)
        sectorarr.append(sector)
        cameraarr.append(camera)
        CCDarr.append(CCD)
        
    lc=np.array(lc).transpose().astype(np.float32)
    tu=np.array(tu).transpose().astype(np.float32)
    asind=np.array(asind).transpose().astype(np.float32)
    bkgu=np.array(bkgu).transpose().astype(np.float32)
    #    infogap=np.array(infogap).transpose().astype(np.int32)

    ntrue=np.array(ntrue).astype(np.uint32)
    tu0=np.array(tu0)
    ##original masked data
#    mask=(t==t)
#    t=t[mask]
#    det=det[mask]
    return lc,tu,asind,bkgu,n,ntrue,nq,inval,tu0,ticarr,sectorarr, cameraarr, CCDarr


def read_tesstic(hdf):

    with h5py.File(hdf,"r") as f:
        
        time=f["LC"]["TIME"].value
        flux=f["LC"]["SAP_FLUX"].value
        q=f["LC"]["QUALITY"].value
        cnts=f["TPF"]["ROW_CNTS"].value

        dta=time[1:]-time[:-1]
        dt = np.median(dta)
        cno = np.round((time-time[0])/dt).astype(np.int32)
        #error=(time - (cno*dt + time[0]) )
        #print(np.max(error))
        
        ra=f["header"]["ra"].value                        
        dec=f["header"]["dec"].value                        
        tic=f["header"]["TID"].value
        sector=f["header"]["sector"].value
        camera=f["header"]["camera"].value
        CCD=f["header"]["chip"].value

        apbkg=f["APERTURE_MASK"]["FLUX_BKG"].value

        bkg=calcbkgd.bkgdlc(cnts,apbkg)
        
        
        return time, flux, q, cno, ra, dec, tic, sector, camera, CCD, cnts, bkg


def throw_tessintarray(n,cno,ndmax,bkg,t,lc,fillvalv=-1.0,fillvalt=-5.0,offt="t[0]"):
    #dt=t[2]-t[1]
    #offt=t[1]    
    offset=np.array(cno[0])
    jend=int(cno[-1]-offset+1)
#    igap=np.ones(n,dtype=np.int)
    lcn=np.ones(n)*fillvalv
    bkgn=np.ones(n)*fillvalv
    
    if(jend > n):
        print("Set larger n than ",jend)
        sys.exit("Error")
#    else:
#        print("Filling ",jend,"-values in ",n," elements in lcn.")
    tun=np.ones(n)*fillvalt
    ndmaxn=np.zeros(n)
    if offt=="t[0]":
        t0=t[0]
    else:
        t0=0.0

    for i in range(0,len(cno)):
        j=int(cno[i]-offset)
        if lc[i]==lc[i] and ndmax[i]==ndmax[i]:    
            lcn[j]=lc[i]
            ndmaxn[j]=ndmax[i]
            tun[j]=t[i]-t0
            bkgn[j]=bkg[i]
#            igap[j]=0
            
    return lcn,tun,ndmaxn,bkgn,t0

    
if __name__ == "__main__":

    mid=1000
    mide=1010
    
    dat=np.load("../data/step3.list.npz")
    filelist=dat["arr_0"][mid:mide]
    rad=dat["arr_1"][mid:mide] #stellar radius
    mass=dat["arr_2"][mid:mide] #stellar mass

    nin=1300
    lc,tu,n,ntrue,nq,inval=load_tesstic(filelist,nin,offt="t[0]",nby=1000)

#    print(lc,n,ntrue)
    
#    for i in range(0,10):
#        t, det, q, cno, ra, dec, tic = read_tesstic(filelist[i])
#        print(tic)
#        n=1500
#        inval=0
#        offt=0.0
#        lcn, tun, t0=throw_tessintarray(n,cno,t,det,fillvalv=1.002,fillvalt=inval,offt=offt)        
#    print(lcn)

