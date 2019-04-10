#/usr/bin/python
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse
from astropy.io import fits
import read_keplerlc as kep
import pandas as pd
from scipy import signal 
import os
from scipy import interpolate

def getkicdir(kicnum,ddir="/sharksuck/kic/data/"):
    #convert kic number (int) to directry name
    strk=str(kicnum)
    if kicnum>99999999:        
        rawdir=strk[0:4]+"/"+strk
    elif kicnum>9999999:        
        rawdir="0"+strk[0:3]+"/0"+strk
    elif kicnum>999999:
        rawdir="00"+strk[0:2]+"/00"+strk        
    else:
        rawdir="000"+strk[0:1]+"/000"+strk
    return ddir+rawdir


def pick_cleaned_lc(kicdir,T0,wid=128,contcrit=84,check=False,lcout=False,tag="",savedir="./"):
    lc,tu,n,ntrue,nq,inval,bjdoffset,t0, t,det=kep.load_keplc([kicdir],offt="0")
    mask=(tu>0.0)
    tuu=tu[mask]
    i=np.searchsorted(tuu,T0)
    numar=np.array(range(0,len(tu)))
    ii=numar[tuu[i]==tu[:,0]][0]

    istart=ii-wid
    iend=ii+wid
    tus=np.copy(tu[istart:iend,0])
    lcs=np.copy(lc[istart:iend,0])

    #pre classifier (check continuous null region)
    prec = True 
    if len(tus) == 0:
        prec=False
    elif tus[0] < 0.0 or tus[-1]<-1:
        contsw=0
        for j, teach in enumerate(tus):
            if teach < 0.0:
                contsw=contsw+1
                if contsw > contcrit:
                    prec=False
            else:
                contsw=0
        
    
    #### CLEAN LCS UP ####
    sw=0
    
    while sw==0:
        tusplus=np.concatenate([[-2000],tus[:-1]])
        tusminus=np.concatenate([tus[1:],[-2000]])
        lcsplus=np.concatenate([[0],lcs[:-1]])
        lcsminus=np.concatenate([lcs[1:],[0]])
        medv=np.median(lcs)

        for j, teach in enumerate(tus):
            if teach < 0.0:
                if tusminus[j] > 0.0 and tusplus[j] > 0.0:
#                    lcs[j] = (lcsminus[j])
                    lcs[j] = (lcsminus[j] + lcsplus[j])/2.0
                    tus[j]=1000.0

                elif tusminus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsminus[j]
                    tus[j]=1000.0
                elif tusplus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsplus[j]
                    tus[j]=1000.0

        if len(tus[tus<0.0])==0:
            sw = 1
    ### median filter
    lcs=signal.medfilt(lcs,kernel_size=3)

    ### NORMALIZE
    lcs=(lcs-np.mean(lcs))/np.std(lcs)
    
    if check:
        fig=plt.figure()
        ax=fig.add_subplot(211)
        istartx=np.max([0,istart-wid])
        iendx=np.min([len(lc),iend+wid])
        lctmp=lc[istartx:iendx]
        tutmp=tu[istartx:iendx]
        mask=(tutmp>0.0)
        ax.plot(tutmp[mask],lctmp[mask],".",color="gray",label="before clean")

        plt.legend()
        ax=fig.add_subplot(212)
        ax.plot(lcs,".",color="orange",label="cleaned vector")
        plt.legend()
        plt.savefig(os.path.join(savedir,tag+"vector.png"))
        #        plt.show()
        
    if lcout:
        return lcs, tus, lc[istart:iend,0], tu[istart:iend,0], lc[:,0], tu[:,0], prec
    else:
        return lcs, tus, prec


def pick_Wnormalized_cleaned_lc(kicdir,T0,W,alpha=2,nx=128,daytopix=48,contcrit=84,check=False,lcout=False,tag="",savedir="./"):
    #nx=64 length of WNC vector
    lc,tu,n,ntrue,nq,inval,bjdoffset,t0, t,det=kep.load_keplc([kicdir],offt="0")
    mask=(tu>0.0)
    tuu=tu[mask]
    i=np.searchsorted(tuu,T0)
    numar=np.array(range(0,len(tu)))
    ii=numar[tuu[i]==tu[:,0]][0]


    wid=int(alpha*W*daytopix)
    print("The range is between -",alpha," W to +",alpha," W." )
    istart=ii-wid
    iend=ii+wid
    tus=np.copy(tu[istart:iend,0])
    lcs=np.copy(lc[istart:iend,0])

    #pre classifier (check continuous null region)
    prec = True 
    if len(tus) == 0:
        prec=False
    elif tus[0] < 0.0 or tus[-1]<-1:
        contsw=0
        for j, teach in enumerate(tus):
            if teach < 0.0:
                contsw=contsw+1
                if contsw > contcrit:
                    prec=False
            else:
                contsw=0
        
    
    #### CLEAN LCS UP ####
    sw=0
    
    while sw==0:
        tusplus=np.concatenate([[-2000],tus[:-1]])
        tusminus=np.concatenate([tus[1:],[-2000]])
        lcsplus=np.concatenate([[0],lcs[:-1]])
        lcsminus=np.concatenate([lcs[1:],[0]])
        medv=np.median(lcs)

        for j, teach in enumerate(tus):
            if teach < 0.0:
                if tusminus[j] > 0.0 and tusplus[j] > 0.0:
#                    lcs[j] = (lcsminus[j])
                    lcs[j] = (lcsminus[j] + lcsplus[j])/2.0
                    tus[j]=1000.0

                elif tusminus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsminus[j]
                    tus[j]=1000.0
                elif tusplus[j] > 0.0:
                    lcs[j] = medv                    
#                    lcs[j] = lcsplus[j]
                    tus[j]=1000.0

        if len(tus[tus<0.0])==0:
            sw = 1
    ### median filter
    lcs=signal.medfilt(lcs,kernel_size=3)

    ### NORMALIZE
    lcs=(lcs-np.mean(lcs))/np.std(lcs)
    nlcs=len(lcs)
    tt=np.array(range(0,nlcs))*2.0*alpha/nlcs - alpha
    fx = interpolate.interp1d(tt, lcs)
    tx=np.array(range(0,nx))*2.0*alpha/nx - alpha
    lcsx=fx(tx)
    
    if check:
        fig=plt.figure()
        ax=fig.add_subplot(211)
        istartx=np.max([0,istart-wid])
        iendx=np.min([len(lc),iend+wid])
        lctmp=lc[istartx:iendx]
        tutmp=tu[istartx:iendx]
        mask=(tutmp>0.0)
        ax.plot(tutmp[mask],lctmp[mask],".",color="gray",label="before clean")

        plt.legend()
        ax=fig.add_subplot(212)
        ax.plot(tt,lcs,".",color="orange")
        ax.plot(tx,lcsx,".",color="red")

        plt.xlabel("time (W)")
        plt.ylabel("WNC vector")
        plt.savefig(os.path.join(savedir,tag+"vector_w.png"))
        #        plt.show()
        
    if lcout:
        return lcs, tus, lc[istart:iend,0], tu[istart:iend,0], lc[:,0], tu[:,0], prec
    else:
        return lcsx, tx, prec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pick up trapezoid candidate for classification (generating cleaned light curves around dips)')
    parser.add_argument('-f', nargs=1, help='info file', type=str)
    parser.add_argument('-n', nargs="+", help='csv file number. needed to use -f. minus value = use all the numbers in f.', type=int)
    parser.add_argument('-nn', nargs=2, help='use -f from n[0] to n[1]', type=int)
    parser.add_argument('-k', nargs=1, default=[1717722], help='kic', type=int)
    parser.add_argument('-t', nargs=1, default=[1439.2],help='time (T0) [BKJD]', type=float)
    parser.add_argument('-w', nargs=1, default=[2.0],help='width [BKJD]', type=float)

    lctag="data"
    mydir="/sharksuck/kic/";
    args = parser.parse_args()
    if args.n:
        nidarr=args.n
        if nidarr[0] < 0:
            dat=pd.read_csv(args.f[0],comment="#")            
            try:
                nidarr=dat["number"].values
            except:
                nidarr=np.asarray(range(0,len(dat)))
    else:
        nidarr=[-1]

    lcsall=[]
    lcsallw=[]
        
    if args.nn:
        dat=pd.read_csv(args.f[0],comment="#")
        try:
            nidarr=dat["number"].values[args.nn[0]:args.nn[1]]
        except:
            nidarr=np.asarray(range(0,len(dat)))[args.nn[0]:args.nn[1]]


    print(nidarr)
    for ii,nid in enumerate(nidarr):
        
        if args.f:        
            dat=pd.read_csv(args.f[0],comment="#")
            if args.n or args.nn:
                try:
                    mask=dat["number"]==nid
                    kicint=dat["KIC"][mask].values[0]
                except:
                    kicint=dat["KIC"].values[ii]
                    mask=dat["KIC"]==kicint
                print("KIC",kicint)
            else:
                kicint=args.k[0]
                mask=dat["KIC"]==kicint
                
            T0=dat["T0BKJD"][mask].values[0]
            W=dat["W"][mask].values[0]
        else:
            kicint=args.k[0]
            T0=args.t[0]
            W=args.w[0]

        ######################################################################
            
        print("*********",kicint,"=",nid,"*********")
        kicdir=getkicdir(kicint,mydir+"data/")+"/"

        lcs, tus, prec=pick_cleaned_lc(kicdir,T0,wid=128,check=True,tag="KIC"+str(kicint),savedir=args.f[0].replace(".txt",""))
        
        if len(lcs[lcs==lcs])>0 and prec:
            lcsall.append(lcs)
            print("1. CLEANED LC IS APPENDED")

        lcsw, tusw, precw=pick_Wnormalized_cleaned_lc(kicdir,T0,W,check=True,tag="KIC"+str(kicint),savedir=args.f[0].replace(".txt",""))
        if len(lcsw[lcsw==lcsw])>0 and precw:
            lcsallw.append(lcsw)
            print("2. WNORMALIZED CLEANED LC IS APPENDED")

            
    np.savez(args.f[0].replace(".txt","")+"_picktrap",lcsall)
    np.savez(args.f[0].replace(".txt","")+"_picktrapW",lcsallw)
        
