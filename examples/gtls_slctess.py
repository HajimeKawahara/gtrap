def injecttransit(lc,tu,nq,mstar,rstar):
    ############# INJECTION #################
    
    #f(P) propto P**-5/3
    xPmin=3.0
    xPmax=30.0    
    alpha=-5/3.
    a1=alpha+1.0
    Y = np.random.random(nq)
    Porb = ((xPmax**(a1) - xPmin**(a1) )*Y + xPmin**(a1))**(1/a1)

    #Radius
    xRpmin=0.2
    xRpmax=1.0    
    Y = np.random.random(nq)    
    Rp = Y*(xRpmax-xRpmin) + xRpmin
    #    Rp= 0.7 ### DEBUG
    
    Mp = 1.0   
    Ms = mstar
    Rs = rstar

    a = (((Porb*u.day)**2 * G * (Ms*M_sun + Mp*M_jup) / (4*np.pi**2))**(1./3)).to(R_sun).value/Rs     
    #b=a*np.cos(ideg/180.0*np.pi)
    b=np.random.random()
    ideg=np.arccos(b/a)/np.pi*180.0
    #print(ideg,"deg")
    #ideg=ideg[0]

    
    tux=np.copy(tu)
    tux[tu<0.0]=None
    tmin=np.nanmin(tux,axis=0)
    tmax=np.nanmax(tux,axis=0)

    
    t0 = np.random.random(nq)*(tmax-tmin)+tmin
    w = 0.0
    e = 0.0
    u1 = 0.1
    u2 = 0.3

    for i in range(0,nq):
        mask=(tu[:,i]>0.0)&(tu[:,i]==tu[:,i])
        ilc, b = gm.gentransit(tu[mask,i].astype(np.float64),t0[i],Porb[i],Rp[i],Mp,Rs[i],Ms[i],ideg[i],w,e,u1,u2)
        if args.q or args.p:
            print("NO INJECTION")
        else:
            lc[mask,i] = lc[mask,i]*(2.0-ilc)

        fac=0.01
        ampS=(Rp[i]/Rs[i])**2*fac*np.nanmean(lc[mask,i])*np.random.random(1)
        ampC=0.5*ampS*np.random.random(1)
        print(Porb[i],t0[i],ampS)
        ilsin=gm.gensin(tu[mask,i].astype(np.float64),Porb[i],t0[i],ampS)
        ilcos=gm.gendcos(tu[mask,i].astype(np.float64),Porb[i],t0[i],ampC)    
        lc[mask,i] = lc[mask,i] + ilsin + ilcos
    return lc

if __name__ == "__main__":
    import astropy.units as u
    from astropy.constants import G, R_sun, M_sun, R_jup, M_jup

    import time
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.compiler
    from pycuda.compiler import SourceModule
    import sys
    import gtrap.gtls as gtls
    import gtrap.gfilter as gfilter
    import gtrap.getstat as getstat
    import gtrap.read_keplerlc as kep
    import h5py
    import argparse
    import sqlite3
    import os
    import gtrap.tls as tls
    import gtrap.detect_peaks as dp
    import sys
    from time import sleep
    import pandas as pd
    import gtrap.genmock as gm
    import gtrap.picktrap as pt
#    import gtrap.read_tesslc as tes
    import gtrap.read_tesstic as tesstic
    import makefig
    
    start = time.time()

    parser = argparse.ArgumentParser(description='GPU Mock/Pick TESS TLS')
    parser.add_argument('-r', help='Randomly selected CTLv3/TIC/', action='store_true')
    parser.add_argument('-i', nargs=1, help='mid start (master ID)', type=int)
    parser.add_argument('-j', nargs=1, help='mid end slice (master ID)', type=int)

    parser.add_argument('-t', nargs='+', help='tic id', type=int)
    parser.add_argument('-m', nargs=1, default=[1],help='Mode: transit=0,lensing=1,absolute=2', type=int)
    parser.add_argument('-o', nargs=1, default=["output.txt"],help='output', type=str)
    parser.add_argument('-n', nargs=1, default=[1],help='Number of picking peaks', type=int)
    parser.add_argument('-fig', help='save figure', action='store_true')
    parser.add_argument('-c', help='Check detrended light curve', action='store_true')
    parser.add_argument('-smt', nargs=1, default=[15],help='smooth', type=int)
    parser.add_argument('-q', help='No injection', action='store_true')
    parser.add_argument('-p', help='picking mode', action='store_true')

    
    ### SETTING
    mpdin = 48 #1 d for peak search margin
    np.random.seed(1)
            
    # #
    args = parser.parse_args()
    mids=args.i[0]
    mide=args.j[0]
    
    dat=np.load("../data/step3.list.npz")
    
    
    ###get filename from the list
    filelist=dat["arr_0"][mids:mide]    
    rstar=dat["arr_1"][mids:mide] #stellar radius
    mstar=dat["arr_2"][mids:mide] #stellar mass
    print("N=",len(dat["arr_0"]))

    nin=2000
    lc,tu,n,ntrue,nq,inval,tu0,ticarr=tesstic.load_tesstic(filelist,nin,offt="t[0]",nby=1000)


    if args.q or args.p:
        print("NO INJECTION")
    else:
        lc=injecttransit(lc,tu,nq,mstar,rstar)
        
    
    ## for transit ##
    if args.m[0] == 0:
        lc = 2.0 - lc
        print("Transit Mode")
    elif args.m[0] == 1:
        print("Lensing Mode")
    elif args.m[0] == 2:
        print("Absolute Mode")
    else:
        sys.exit("No mode")
    ###############

    

    #median filter width
    medsmt_width = 32
    nby=1000 ## of thread
    dev_imgout=gfilter.get_detrend(lc,nby=nby,nbx=1,r=medsmt_width,isw=0,osw=1) #detrend
    
    #set
    tu=np.array(tu,order="C").astype(np.float32)

    #determine deltat
    ttmp=tu[:,0]
    numl=np.array(list(range(0,len(ttmp))))
    mask=(ttmp>=0.0)
    deltat=(ttmp[mask][-1] - ttmp[mask][0])/(numl[mask][-1] - numl[mask][0])

    #tbls setting
    #the number of the window width= # of the threads should be 2^n !!!!
    nw=1024 
    
    # Max and Min of Widths
    wmin = 0.2  #  day
    wmax = 1.0  # day
    dw=(wmax-wmin)/(nw-1)
    t0min=(2*wmin) #test

    dt=1.0
    nt=len(tu[:,0])
    #L
    nl=20

    # the number of a scoop
    nsc=int(wmax/deltat+3.0)

    print(nl,nt,nsc,nw)
    print("# of threads (W) = ",nw)
    print("# of for-loop (L) = ",nl)
    print("scoop number = ",nsc)
    
    
    #    imgout=np.array(imgout,order="F").astype(np.float32)
    #lc=np.copy(imgout)
    #dev_ntrue = cuda.mem_alloc(ntrue.nbytes)
    #cuda.memcpy_htod(dev_ntrue,ntrue)

    dev_tu = cuda.mem_alloc(tu.astype(np.float32).nbytes)
    cuda.memcpy_htod(dev_tu,tu.astype(np.float32))

    ntrue=ntrue.astype(np.int32)
    dev_ntrue = cuda.mem_alloc(ntrue.nbytes)
    cuda.memcpy_htod(dev_ntrue,ntrue)
    
    #output TLS s/n, w, l
    tlssn=np.zeros(nt*nq).astype(np.float32)
    dev_tlssn = cuda.mem_alloc(tlssn.nbytes)

    tlsw=np.zeros(nt*nq).astype(np.float32)
    dev_tlsw = cuda.mem_alloc(tlsw.nbytes)

    tlst0=np.zeros(nt*nq).astype(np.float32)
    dev_tlst0 = cuda.mem_alloc(tlst0.nbytes)

    tlsl=np.zeros(nt*nq).astype(np.float32)
    dev_tlsl = cuda.mem_alloc(tlsl.nbytes)

    tlshmax=np.zeros(nt*nq).astype(np.float32)
    dev_tlshmax = cuda.mem_alloc(tlshmax.nbytes)

    source_module=gtls.gtls_module()
    
    ##compute kma,kmi,kkmi
    sharedsize=(2*nsc + nw + 2)*4 #byte
    print("sharedsize=",sharedsize)
    #gtls
#    start = time.time()
    if args.m[0] == 2:
        source_module=gtls.gtls_module("absolute")
    else:
        source_module=gtls.gtls_module()
    pkernel=source_module.get_function("gtls")

    pkernel(dev_tlssn,dev_tlsw,dev_tlst0,dev_tlsl,dev_tlshmax,\
            #dev_debugt,dev_debuglc,dev_debugpar,\
            dev_imgout,dev_tu,\
            np.int32(nt),\
            np.int32(nl),np.int32(nsc),\
            np.float32(t0min),np.float32(dt),\
            np.float32(wmax),np.float32(dw),np.float32(deltat),\
            block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)
        
    cuda.memcpy_dtoh(tlssn, dev_tlssn)
    cuda.memcpy_dtoh(tlsw, dev_tlsw)
    cuda.memcpy_dtoh(tlst0, dev_tlst0)
    cuda.memcpy_dtoh(tlsl, dev_tlsl)
    cuda.memcpy_dtoh(tlshmax, dev_tlshmax)

    #========================================--

    ########################
    PickPeaks=args.n[0]
    for iq,tic in enumerate(ticarr):
        detection=0
        lab=0
        fac=1.0
        ffac = 8.0 #region#

        mask = (tlssn[iq::nq]>0.0)        
        std=np.std(tlssn[iq::nq][mask])
        median=np.median(tlssn[iq::nq])
        #### PEAK STATISTICS ####
        peak = dp.detect_peaks(tlssn[iq::nq],mpd=mpdin)
        peak = peak[np.argsort(tlssn[iq::nq][peak])[::-1]]        
        
        PickPeaks=min(PickPeaks,len(tlssn[iq::nq][peak]))
        print("Pick PEAK=",PickPeaks)
        
        maxsn=tlssn[iq::nq][peak][0:PickPeaks]
        Pinterval=np.abs(tlst0[iq::nq][peak][1]-tlst0[iq::nq][peak][0])
        far=(maxsn-median)/std

        minlen =  10000.0 #minimum length for time series
        lent =len(tlssn[iq::nq][tlssn[iq::nq]>0.0])
        idetect = peak[0]
        for ipick in range(0,PickPeaks):
            print("##################"+str(ipick)+"##########################")

            if True:

                i = peak[ipick]
                im=np.max([0,int(i-nsc*ffac)])
                ix=np.min([nt,int(i+nsc*ffac)])
                imn=np.max([0,int(i-nsc/2)])
                ixn=np.min([nt,int(i+nsc/2)])
                
            
                if args.m[0] == 0:    
                    llc=2.0 - lc[im:ix,iq]
                    llcn=2.0 - lc[imn:ixn,iq]
                    
                elif args.m[0] == 1 or args.m[0] == 2:
                    llc=lc[im:ix,iq]
                    llcn=lc[imn:ixn,iq]
            
                ttc=tu[im:ix,iq]
                ttcn=tu[imn:ixn,iq]#narrow region
                
                #PEAK VALUE
                T0=tlst0[iq::nq][peak[ipick]]+tu0[iq]
                W=tlsw[iq::nq][peak[ipick]]
                L=tlsl[iq::nq][peak[ipick]]
                H=tlshmax[iq::nq][peak[ipick]]
                
                #################
                print("GPU ROUGH: T0,W,L,H")
                print(T0,W,L,H)

                if args.q or args.p:
                    dTpre=0.0
                else:
                    dTpre=np.abs((np.mod(T0,Porb) - np.mod(t0+tu0[iq],Porb))/(W/2))
                print("DIFF/dur=",dTpre)

                if (dTpre < 0.1 and detection == 0) or args.q or args.p:
                    #print("(*_*)/ DETECTED at n=",ipick+1," at ",peak[ipick])

                    detection = ipick+1
                    idetect = i
                    if args.q:
                        lab=0
                    elif args.p:
                        lab=-1
                    else:
                        lab=1
                                   

                    T0tilde=tlst0[iq::nq][peak[ipick]]
                    ## REINVERSE
                    if args.m[0] == 0:
                        lc = 2.0 - lc

                    if True:
#                    try:
                        lcs, tus, prec=pt.pick_Wnormalized_cleaned_lc_direct(lc,tu,T0tilde,W,alpha=1,nx=201,check=True,tag="TIC"+str(tic)+"s"+str(lab),savedir="mocklc_slctess")      
                        lcsw, tusw, precw=pt.pick_Wnormalized_cleaned_lc_direct(lc,tu,T0tilde,W,alpha=3,nx=2001,check=True,tag="TIC"+str(tic)+"w"+str(lab),savedir="mocklc_slctess")
                        #                print(len(lcs),len(lcsw))
                        starinfo=[mstar,rstar]
                        if args.p:
                            np.savez("picklc_slctess/pick_slctess"+str(tic)+"_"+str(ipick),[lab],lcs,lcsw,starinfo)
                        else:
                            np.savez("mocklc_slctess/mock_slctess"+str(tic)+"_"+str(ipick),[lab],lcs,lcsw,starinfo)


    elapsed_time = time.time() - start
    print (("2 :{0}".format(elapsed_time)) + "[sec]")
    print (elapsed_time/(mide-mids), "[sec/N]")

                            #                    except:
#                        print("SOME ERROR for SAVE")
                    ###############################################
#                    if args.fig:
#                        makefig.trapfig(tlst0[iq::nq],tlssn[iq::nq],tlsw[iq::nq],tu[im:ix,iq],tu0[iq],llc,peak,idetect,ttc,ttcn,llcn,offsetlc,ffac,H,W,L,T0,args.m[0],tic)
            
#        ff = open("mockslc_checkoutput."+str(medsmt_width)+".txt", 'a')
#        ff.write(str(tic)+","+str(detection)+"\n")
#        ff.close()
#        print(tic)
#            plt.savefig("KIC"+str(kic)+".pdf", bbox_inches="tight", pad_inches=0.0)
#            plt.show()

