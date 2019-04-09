if __name__ == "__main__":
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
    import tls
    import detect_peaks as dp
    import sys
    from time import sleep
    import gpucheck
    start = time.time()

    parser = argparse.ArgumentParser(description='GPU Kepler TLS')
    parser.add_argument('-i', nargs='+', help='sharksuck id', type=int)
    parser.add_argument('-k', nargs='+', help='kic id', type=int)
    parser.add_argument('-t', nargs=1, help='testkoi id', type=int)
    parser.add_argument('-q', nargs=1, default=["BLS_TEST"],help='SQL table name', type=str)
    parser.add_argument('-d', nargs=1, default=["bls.db"],help='SQL db name', type=str)
    parser.add_argument('-b', nargs=1, default=[0],help='batch num', type=int)
    parser.add_argument('-f', nargs=1, default=["fits"],help='filetype: fits (BKJD), hdf5 (relative Day)', type=str)
    parser.add_argument('-m', nargs=1, default=[0],help='Mode: transit=0,lensing=1,absolute=2', type=int)
    parser.add_argument('-o', nargs=1, default=["output.txt"],help='output', type=str)
    parser.add_argument('-n', nargs=1, default=[1],help='The target of the number of the transits: STE=1, DTE=2 for the max SN. ', type=int)
    #    parser.add_argument('-p', nargs=1, default=[400],help='Minimum interval for DTE (d)', type=float)
    parser.add_argument('-fig', help='save figure', action='store_true')
    parser.add_argument('-gpu', nargs=3, default=[85.0,51.0,30.0],help='GPU control, limit temperature [K], settle temperature, sleep time [sec] for cool down', type=float)

    ### SETTING
    mpdin = 48 #1d for peak search margin
            
    # #
    args = parser.parse_args()

    #### GPU HEALTH CHECK #######
    maxt=args.gpu[0]
    sett=args.gpu[1]
    sleeptime=args.gpu[2]
    maxgput=gpucheck.getmaxgput_gput()
    print(maxgput,"[C] for init")
    if maxgput > maxt:
        print("Fever. Waiting for cool down of the fevered GPU. ")
        fever = True
        while fever :
            sleep(sleeptime)
            maxgput=gpucheck.getmaxgput_gput()
            print(maxgput,"[C]")
            if maxgput < sett:
                fever = False
    #### SLEEP TIME #######


    sqltag=args.q[0]
    blsdb=args.d[0]

    sidlist=[]
    kicintlist=[]
    if args.i:
        sidlist=args.i
        conn=sqlite3.connect("/sharksuck/kic/kic.db")
        cur=conn.cursor()
        for sid in sidlist:
            cur.execute("SELECT kicint FROM kic where id="+str(sid)+";")
            kicint=cur.fetchone()[0]
            kicintlist.append(kicint)
            conn.commit()
            print("KICINT=",kicint)
        conn.close()

    elif args.k:
        conn=sqlite3.connect("/sharksuck/kic/kic.db")
        cur=conn.cursor()
        kicintlist=args.k
        for kicint in kicintlist:
            cur.execute("SELECT id FROM kic where kicint="+str(kicint)+";")
            sid=cur.fetchone()[0]
            sidlist.append(sid)
            conn.commit()
            print("SID=",sid)
        conn.close()

        
    
    #### KICDIR
    kicdirlist=[]
    conn=sqlite3.connect("/sharksuck/kic/kic.db")
    cur=conn.cursor()
    for sid in sidlist:
        cur.execute("SELECT kicdir FROM kic where id="+str(sid))
        kicdirlist.append(cur.fetchone()[0])
    conn.commit()
    conn.close()


    #########################################################
    print("###################################################")
    print("# WARNING:BRIGHTNESS LIMIT #")
    print("###################################################")

    dat=np.load("data/rmag.npz")
    kicintar=np.array(dat["arr_0"][:,0]).astype(np.int)
    rmagar=np.array(dat["arr_0"][:,1])
    ##############################################################
        
    
    lc=[]
    tu=[]
    n=len(kicdirlist)

    if args.f[0]=="hdf5":
        print("HDF5 mode, relative day.")
        tu0=[]
        for i,kicdir in enumerate(kicdirlist):
            path=kicdir.replace("data",sqltag)
            if not os.path.exists(path):
                os.makedirs(path)

            infile = os.path.join(kicdir,str(kicintlist[i])+".h5")
            h5file = h5py.File(infile,"r")
            lc.append(h5file["/lc"].value)
            tu.append(h5file["/tu"].value)

            par  = h5file["/params"].value 
            n,ntrue,nq,inval=np.array(par,dtype=np.int)
            h5file.flush()
            h5file.close()
            tu0.append(0.0)#????

        tu=np.array(tu).transpose()
        lc=np.array(lc).transpose()
        
    elif args.f[0]=="fits":
        print("fits mode, BKJD.")        
        lc,tu,n,ntrue,nq,inval,bjdoffset,t0arr, t, det=kep.load_keplc(kicdirlist)
        tu0=t0arr
    else:
        sys.exit("No file type -f "+args.f[0])
    print("number of the KICs, nq=",nq)
    print("offset=",tu0)
    ##OFFSET


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
    nby=1000 ## of thread
    dev_imgout=gfilter.get_detrend(lc,nby=nby,nbx=1,r=128,isw=0,osw=1) #detrend
    
    #set
    tu=np.array(tu,order="C").astype(np.float32)

    #determine deltat
    ttmp=tu[:,0]
    numl=np.array(list(range(0,len(ttmp))))
    mask=(ttmp>=0.0)
    deltat=(ttmp[mask][-1] - ttmp[mask][0])/(numl[mask][-1] - numl[mask][0])

    #tbls setting
    #the number of the window width= # of the threads should be 2^n !!!!
    nw=128 
    
    # Max and Min of Widths
    wmin = 1.0  #  day
    wmax = 6.0  # day
    dw=(wmax-wmin)/(nw-1)
    t0min=(2*wmin) #test
    t0max=n-2*wmin #test

    dt=1.0
    nt=len(tu[:,0])
    #L
    nl=10

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
    dat=np.load("data/cdpp15.npz")
    kicintccdp=np.array(dat["arr_0"][:,0]).astype(np.int)
    ccdp15=np.array(dat["arr_0"][:,1])

    ########################
    PeakMaxindex=args.n[0]-1
    

    for iq,kic in enumerate(kicintlist):

        ind=np.searchsorted(kicintar,kic)
        rmag=(rmagar[ind])

        fac=1.0
        ffac = 8.0 #region#


        mask = (tlssn[iq::nq]>0.0)        
        std=np.std(tlssn[iq::nq][mask])
        median=np.median(tlssn[iq::nq])
        #### PEAK STATISTICS ####
        peak = dp.detect_peaks(tlssn[iq::nq],mpd=mpdin)
        peak = peak[np.argsort(tlssn[iq::nq][peak])[::-1]]        
        ntop = 4 # the nuber of the top peaks
        
        if len(peak) > ntop:

        ### OLD ###
#            stdc=np.nanstd(tlssn[iq::nq][peak[ntop:]]) #peak std
#            medc=np.nanmedian(tlssn[iq::nq][peak[ntop:]])
#            pssn=(tlssn[iq::nq][peak[0]]-medc)/stdc

            i = np.argmax(tlssn[iq::nq])
            ind=np.searchsorted(kicintccdp,kic)
            ccdpval=ccdp15[ind]*1.e-6
            pssn = tlshmax[iq::nq][i]/ccdpval
            
        else:
            pssn = np.inf

#===========CRIT 1==========================       
        if len(peak) > ntop:
            peak=peak[0:ntop]
            peakval = tlssn[iq::nq][peak]
            print(peakval)
            ## dntop-1-th peak/std
            diff=(peakval[-1]-median)/std
        else:
            ntop=len(peak)            
            diff = 0.0
#===========================================
        
        maxsn=tlssn[iq::nq][peak][PeakMaxindex]
        Pinterval=np.abs(tlst0[iq::nq][peak][1]-tlst0[iq::nq][peak][0])
        far=(maxsn-median)/std

        minlen =  10000.0 #minimum length for time series
        lent =len(tlssn[iq::nq][tlssn[iq::nq]>0.0])

#        if maxsn > 1.8:
#            xxcrit=10.5*np.log10(maxsn-1.8)**0.6-3.5
#            #xxcrit=10.5*np.log10(maxsn-1.8)**0.6-1.5
#            xxcrit=10.5*np.log10(maxsn-1.8)**0.6-0.9 #DTE
#        else:
#            xxcrit=0.0

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
            T0p=tlst0[iq::nq][peak][0]+tu0[iq]
            Wp=tlsw[iq::nq][peak][0]
            Lp=tlsl[iq::nq][peak][0]
            Hp=tlshmax[iq::nq][peak][0]

            #################
            print("GPU ROUGH: T0,W,L,H")
            print(T0p,Wp,Lp,Hp)
            ### RE (PRECISE) FIT
            NT0=10
            dT0=0.1
            tl=np.linspace(T0p-dT0,T0p+dT0,NT0)
            NW=10
            dW=Wp/100
            Wl=np.linspace(Wp-dW,Wp+dW,NW)
            NL=20
            Lpd=0.0
            Lpu=Wp/2.0
            Ll=np.linspace(Lpd,Lpu,NL)
            detpre,H,W,L,T0,offsetlc=tls.tlsfit(t,det,Wl,Ll,tl)
            H=-H
            print("PRECISE TRAFIT: T0,W,L,H")
            print(T0,W,L,H)
            
            if args.o:
                ttag=args.o[0].replace(".txt","")
            else:
                ttag="_"

            ofile=args.o[0]
            f = open(ofile, 'a')
            f.write(str(kic)+","+str(maxsn)+","+str(far)+","+str(diff)+","+str(pssn)+","+str(lent)+","+str(Pinterval)+","+str(H)+","+str(tlst0[iq::nq][i]+tu0[iq])+","+str(L)+","+str(W)+"\n")
            f.close()

                
            ff = open("steinfo"+ttag+".txt", 'a')
            ff.write(str(kic)+","+str(H)+","+str(tlst0[iq::nq][i]+tu0[iq])+","+str(L)+","+str(W)+",\n")
            ff.close()
            
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
                plt.title("KIC"+str(kic)+" Pi="+str(Pinterval),fontsize=18)
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
                
                ax=fig.add_subplot(3,1,3)
                plt.tick_params(labelsize=18)
                
                pre=tls.gen_trapzoid(ttc[mask]+tu0[iq],H,W,L,T0)
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

                plt.savefig("KIC"+str(kic)+".png")
                print("t0=",tlst0[iq::nq][i]+tu0[iq])
                print("height=",H)
                print("L=",L)



#            plt.savefig("KIC"+str(kic)+".pdf", bbox_inches="tight", pad_inches=0.0)
#            plt.show()

