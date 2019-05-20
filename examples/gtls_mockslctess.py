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
    
    start = time.time()

    parser = argparse.ArgumentParser(description='GPU Mock TESS TLS')
    parser.add_argument('-r', help='Randomly selected CTLv3/TIC/', action='store_true')
    parser.add_argument('-i', nargs=1, help='mid (master ID)', type=int)
    parser.add_argument('-t', nargs='+', help='tic id', type=int)
    parser.add_argument('-m', nargs=1, default=[1],help='Mode: transit=0,lensing=1,absolute=2', type=int)
    parser.add_argument('-o', nargs=1, default=["output.txt"],help='output', type=str)
    parser.add_argument('-n', nargs=1, default=[1],help='Number of picking peaks', type=int)

    parser.add_argument('-fig', help='save figure', action='store_true')
    parser.add_argument('-c', help='Check detrended light curve', action='store_true')
    parser.add_argument('-smt', nargs=1, default=[15],help='smooth', type=int)

    ### SETTING
    mpdin = 48 #1/3d for peak search margin
            
    # #
    args = parser.parse_args()


    ###get filename from the list
    mid = args.i[0]
    dat=np.load("../data/step3.list.npz")
    fileone=dat["arr_0"][mid]    
    rstar=dat["arr_1"][mid] #stellar radius
    mstar=dat["arr_2"][mid] #stellar mass


    nlc=2001
    t, det, q, cno, ra, dec, ticint = tesstic.read_tesstic(fileone)

    #masking quality flaged bins
    mask=(q==0)
    t=t[mask]
    det=det[mask]
    cno=cno[mask]

    #set arrayed bin
    lc,tu, tu0 = tesstic.throw_tessintarray(nlc,cno,t,det,fillvalv=-1.0,fillvalt=-5.0,offt="t[0]")
    
    lc=np.array([lc]).transpose()
    tu=np.array([tu]).transpose()
    gapmask=(tu<0)
    ntrue=np.array([len(tu[~gapmask])])
    nq=1
    tu0=np.array([tu0]) ##should change
    ############# INJECTION #################
    
    #f(P) propto P**-5/3
    xPmin=3.0
    xPmax=30.0    
    alpha=-5/3.
    a1=alpha+1.0
    Y = np.random.random()
    Porb = ((xPmax**(a1) - xPmin**(a1) )*Y + xPmin**(a1))**(1/a1)

    #Radius
    xRpmin=0.2
    xRpmax=1.0    
    Y = np.random.random()    
    #Rp = Y*(xRpmax-xRpmin) + xRpmin
    Rp= 1.0 ### DEBUG
    Mp = 1.0
    
    Ms = mstar
    Rs = rstar

    a = (((Porb*u.day)**2 * G * (Ms*M_sun + Mp*M_jup) / (4*np.pi**2))**(1./3)).to(R_sun).value/Rs     
    #b=a*np.cos(ideg/180.0*np.pi)
    b=np.random.random()
    ideg=np.arccos(b/a)/np.pi*180.0
    #print(ideg,"deg")
    #ideg=ideg[0]
    print(ideg,"deg")
    
    mask=(tu>0.0)&(tu==tu)

    tmin=np.nanmin(tu[mask])
    tmax=np.nanmax(tu[mask])

    t0 = np.random.random()*(tmax-tmin)+tmin
    w = 0.0
    e = 0.0
    u1 = 0.1
    u2 = 0.3
    
    ilc, b = gm.gentransit(tu[mask].astype(np.float64),t0,Porb,Rp,Mp,Rs,Ms,ideg,w,e,u1,u2)
    lc[mask] = lc[mask]*(2.0-ilc)

    if args.fig:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.axvline(t0)
        ax.plot(tu,lc,".")
        plt.ylim(np.min(lc[lc>0.0]),np.max(lc))
        ax = fig.add_subplot(212)
        ax.axvline(t0)
        ax.plot(tu,lc,".")
        plt.ylim(np.min(lc[lc>0.0]),np.max(lc))
        plt.xlim(t0-1,t0+1)
        plt.savefig("test.png")
    #########################################
    
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
    
    elapsed_time = time.time() - start
    print (("2 :{0}".format(elapsed_time)) + "[sec]")

    ##detrend (directly use)

    #median filter width
    medsmt_width = 32


    nby=1000 ## of thread
    print(lc)

    
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
    start = time.time()
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
    detection=0
    lab=0
    for iq,tic in enumerate([ticint]):

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

        for ipick in range(0,PickPeaks):
            print("##################"+str(ipick)+"##########################")

            i = peak[ipick]
    
            if True:
                #        if (diff < xxcrit and diff < 8.0 and float(lent) > minlen and Pinterval > 300.0) or args.n[0]==1:
                i = np.argmax(tlssn[iq::nq])
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
                xmask=(t-T0>=-W/2)*(t-T0<=W/2)
                offsetlc=np.nanmean(det[xmask])
                
                dTpre=np.abs((np.mod(T0,Porb) - np.mod(t0+tu0[0],Porb))/(W/2))
                print("DIFF/dur=",dTpre)

                if dTpre < 0.2:
                    detection = ipick+1
                    lab=1
                                   
                if True:
#            if dTpre < 0.25: 
                    if args.o:
                        ttag=args.o[0].replace(".mock_slctess.txt","")
                    else:
                        ttag="_"                
                                

                    T0tilde=tlst0[iq::nq][i]#+tu0[iq]
                    ## REINVERSE
                    if args.m[0] == 0:
                        lc = 2.0 - lc
                        
                        #                lcs, tus, prec=pt.pick_cleaned_lc_direct(lc,tu,T0tilde,wid=100,check=True,tag="KIC"+str(kicint),savedir="mocklc")
                        
                    lcs, tus, prec=pt.pick_Wnormalized_cleaned_lc_direct(lc,tu,T0tilde,W,alpha=1,nx=51,check=True,tag="TIC"+str(tic)+"s"+str(lab),savedir="mocklc_slctess")      
                    lcsw, tusw, precw=pt.pick_Wnormalized_cleaned_lc_direct(lc,tu,T0tilde,W,alpha=5,nx=201,check=True,tag="TIC"+str(tic)+"w"+str(lab),savedir="mocklc_slctess")
                    #                print(len(lcs),len(lcsw))
                    starinfo=[mstar,rstar]                                   
                    np.savez("mocklc_slctess/mock_slctess"+str(tic),[lab],lcs,lcsw,starinfo)
            
            ###############################################

            
            if args.fig:
                fig=plt.figure(figsize=(10,7.5))
                ax=fig.add_subplot(3,1,1)
                plt.tick_params(labelsize=18)
                
                plt.plot(tlst0[iq::nq]+tu0[iq],tlssn[iq::nq],color="gray")
                plt.plot(tlst0[iq::nq][peak]+tu0[iq],tlssn[iq::nq][peak],"v",color="red",alpha=0.3)
                plt.axvline(tlst0[iq::nq][i]+tu0[iq],color="red",alpha=0.3)
                plt.axhline(median,color="green",alpha=0.3)
                plt.ylabel("TLS series",fontsize=18)            
                #            plt.title("KIC"+str(kic)+" SN="+str(round(maxsn,1))+" far="+str(round(far,1))+" dp="+str(round(diff,1)))
                plt.title("TIC"+str(tic)+" (transit injected) Pi="+str(Pinterval),fontsize=18)
                plt.axvline(t0+tu0,color="orange",ls="dotted")

                ax=fig.add_subplot(3,1,2)
                plt.tick_params(labelsize=18)
                
                plt.xlim(tlst0[iq::nq][i]-tlsw[iq::nq][i]*ffac+tu0[iq],tlst0[iq::nq][i]+tlsw[iq::nq][i]*ffac+tu0[iq])
                ax.plot(tu[im:ix,iq]+tu0[iq],llc,".",color="gray")
                
                ##==================##
                
                mask=(ttc>0.0)
                dip=np.abs((np.max(llc[mask])-np.min(llc[mask]))/3.0)
                maskn=(ttcn>0.0)
                
                plt.ylim(np.min(llcn[maskn])-dip,np.max(llcn[maskn])+dip)
                plt.ylabel("PDCSAP",fontsize=18)
                plt.axvline(t0,color="red")
                
                ax=fig.add_subplot(3,1,3)
                plt.tick_params(labelsize=18)
                
                pre=tls.gen_trapzoid(ttc[mask]+tu0[iq],H,W,L,T0)
#                pre=tls.gen_trapzoid(ttc[mask]+tu0[iq],Hp,Wp,Lp,T0p)
                if args.m[0] == 0:    
                    pre = 1.0 - pre
                elif args.m[0] == 1:    
                    pre = 1.0 + pre
                ax.plot(ttc[mask]+tu0[iq],pre-(1.0-offsetlc),color="green")
                ax.plot(ttc[mask]+tu0[iq],pre-(1.0-offsetlc),".",color="green",alpha=0.3)

                plt.xlim(tlst0[iq::nq][i]-tlsw[iq::nq][i]*ffac+tu0[iq],tlst0[iq::nq][i]+tlsw[iq::nq][i]*ffac+tu0[iq])
                plt.ylim(np.min(llcn[maskn])-dip,np.max(llcn[maskn])+dip)
                plt.ylabel("best matched",fontsize=18)
                if tu0[iq]==0.0:
                    plt.xlabel("Day",fontsize=18)
                else:
                    plt.xlabel("BKJD",fontsize=18)

                plt.axvline(t0,color="red")
                    
                plt.savefig("TIC"+str(tic)+".mock.png")
                print("t0=",tlst0[iq::nq][i]+tu0[iq])
                print("height=",H)
                print("L=",L)


        ff = open("mockslc_checkoutput."+str(medsmt_width)+".txt", 'a')
        ff.write(str(tic)+","+str(detection)+"\n")
        ff.close()

#            plt.savefig("KIC"+str(kic)+".pdf", bbox_inches="tight", pad_inches=0.0)
#            plt.show()

