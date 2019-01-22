import time
import matplotlib.pyplot as plt
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule

def gtls_module ():
    source_module = SourceModule("""

    #include <stdio.h>
    #define FILLVAL -1.0

    /* criterion for the filling factor of the data for a trapezoid */
    #define FILFACCRIT 0.7

    extern __shared__ float cache[]; 


/* each thread = W shift */

    __global__ void gtls(float *tlssn, float *tlsw, float *tlst0, float *tlsl, float *tlshmax, float *lc, float *tu, int nt, int nl, int nsc, float t0min, float dt, float wmax, float dw, float deltat){ 

/*  
    +batch structure
    batch number = blockIdx.y

    +structure of cache 
    cache[nsc], lc in a scoop: cache[j]  for j=0,...,nsc-1 
    cache[nsc], tu - t0 in a scoop: cache[j]  for j=nsc,...,2*nsc-1 
    cache[nthread], res_max along the width (W) direction for j=2*nsc,....,2nsc+nthread-1
    cache[1], sum of lc i.e. j=2*nsc + nthread

*/

    int nthread = blockDim.x;
    int ithread =  threadIdx.x;

    float rnsc=float(nsc);
    float rnthread=float(nthread);
    int i;
    int j;

/* ===================================================== */
/* thread cooperating initialization (1) */    
/* The default value of t should be negative (FILLVAL) */

    for (int m=0; m<int(rnsc/rnthread)+1; m++){
         i = m*nthread+ithread;
         if (i < nsc){ 
              cache[i]=0.0;
              cache[i+nsc]=FILLVAL;
         }

    }

    if (ithread < nthread){ 
    cache[ithread+2*nsc]=0.0;
         }

    if(ithread==0){
    cache[2*nsc+nthread]=0.0;
    cache[2*nsc+nthread+1]=0.0;
    }

    __syncthreads();
/* ===================================================== */


/* thread cooperating reading a scoop */
    int k = blockIdx.x*gridDim.y+blockIdx.y;
    int nsch=int(nsc/2);
   
    for (int m=0; m<int(rnsc/rnthread)+1; m++){

    i = m*nthread+ithread;
    j = k+2*(i - nsch);
    if (i >= 0 && i < nsc && j < 2*nt && j >= 0){
    /* C-wise */
        atomicAdd(&cache[i],lc[j]);  
        atomicAdd(&cache[i+nsc],tu[j]-FILLVAL); 
    }
                
    }

    /* - Compute chisq */
    float W = wmax - dw*float(ithread); 
    float dl = W/4.0/float(nl);
    float L = 0.0;
    float Lmax = 0.0;
    float Hmaxmax = 0.0;

    float res;
    float res_max;
    float ele;
    float xAEC=0.0;
    float xBD=0.0;
    float xtB=0.0;
    float xtD=0.0;
    float t2BD=0.0;
    float tB=0.0;
    float tD=0.0;

    float xtmp=0.0;
    float tStart;
    float tEnd;
    float tAB;
    float tBC;
    float tCD;
    float tDE;

    /* non mask n */
    int nSx=0;
    int nAx=0;
    int nBx=0;
    int nCx=0;
    int nDx=0;
    int nEx=0;

    /* masked n */
    int nA=0;
    int nB=0;
    int nC=0;
    int nD=0;
    int nE=0;

    float tnow;
    float denA;
    float numB;
    float Hmax;
    float Hmaxh;

    res_max=0.0;

    tEnd = 0.5*W;
    tStart = - 0.5*W;

    __syncthreads();

    /* shifting time */
    for (int m=0; m<int(rnsc/rnthread)+1; m++){

    i = m*nthread+ithread;

    if( i >= 0 && i < nsc && cache[nsc+i] >= 0.0){
    cache[i+nsc] = cache[i+nsc] - deltat*float(blockIdx.x);
    }else if(i>=0.0 && i < nsc && cache[nsc+i] < 0.0){
    /* NO NEED for FAST VERSION */
    cache[i+nsc] = -1000.0;
    cache[i] = 0.0;
    }
                
    }

    __syncthreads();

/* get mean for each thread */
    float mean=0.0;
    float nsum=0.0;
    for (int m=0; m<nsc; m++){

    if(cache[m+nsc]>=tStart && cache[m+nsc]<=tEnd){
    mean = mean + cache[m];
    nsum = nsum + 1.0;
    }

    }

    if(nsum>0){
    mean=mean/nsum;
    }

        
    __syncthreads();


    /* =============================== */
    /* main loop for l */
    /* =============================== */

    for (int mm=0; mm<nl; mm++){
    L = float(mm+1)*dl;

    xAEC=0.0;
    xBD=0.0;
    xtB=0.0;
    xtD=0.0;
    t2BD=0.0;
    tB=0.0;
    tD=0.0;

    xtmp=0.0;
    nA=0;
    nB=0;
    nC=0;
    nD=0;
    nE=0;
    nSx=0;
    nAx=0;
    nBx=0;
    nCx=0;
    nDx=0;
    nEx=0;

    /* -- boundary */
    tAB = - W*0.25 - L*0.5;
    tBC = - W*0.25 + L*0.5;
    tCD = - tBC;
    tDE = - tAB;
    
    /* ============================================== */
    /* -- search the starting -- */

    for (int i=0; i<nsc; i++){
      
    if(cache[i+nsc] > tStart){
    break;
    }
    nSx=nSx+1;

    }


    /* ============================================== */
    /* -- search the region A -- */

    for (int i=0; i<nsc-nSx; i++){

    tnow=cache[i+nSx+nsc];
    if(tnow > tAB){
    break;
    }

    if(tnow >= tStart){
    nA = nA+1;
    xAEC = xAEC + cache[i] - mean;
    }
    nAx=nAx+1;

    }


    /* ============================================== */
    /* -- search the region B -- */

    for (int i=0; i<nsc-nAx-nSx; i++){

    tnow=cache[i+nSx+nAx+nsc];

    if(tnow > tBC){
    break;
    }

    if(tnow >= tStart){
    nB = nB+1;
    xtmp=cache[i+nSx+nAx] - mean;
    xBD = xBD + xtmp;
    tB = tB + tnow;
    xtB = xtB + xtmp*tnow;
    t2BD = t2BD + tnow*tnow; 
    }
    nBx=nBx+1;

    }

    /* ============================================== */
    for (int i=0; i<nsc-nAx-nBx-nSx; i++){

    tnow=cache[i+nSx+nAx+nBx+nsc];

    if(tnow > tCD){
    break;
    }

    if(tnow >= tStart){
    nC = nC+1;
    xAEC = xAEC + cache[i+nSx+nAx+nBx] - mean;
    }
    nCx=nCx+1;



    }

    /* ============================================== */
    for (int i=0; i<nsc-nSx-nAx-nBx-nCx; i++){

    tnow=cache[i+nSx+nAx+nBx+nCx+nsc];
    if(tnow > tDE){
    break;
    }

    if(tnow >= tStart){
    nD = nD+1;
    xtmp=cache[i+nSx+nAx+nBx+nCx] - mean;
    xBD = xBD + xtmp;
    tD = tD + tnow;
    xtD = xtD + xtmp*tnow;
    t2BD = t2BD + tnow*tnow; 
    }
    nDx=nDx+1;

    }

    /* ============================================== */
    
    for (int i=0; i<nsc-nAx-nBx-nCx-nDx-nSx; i++){

    tnow=cache[i+nSx+nAx+nBx+nCx+nDx+nsc];
    if(tnow > tEnd){
    break;
    }

    if(tnow >= tStart){
    nE = nE+1;
    xAEC = xAEC + cache[i+nSx+nAx+nBx+nCx+nDx] - mean;
    }

    nEx=nEx+1;
    }

    /* filling factor of the data */
    float filfac = float(nA+nB+nC+nD+nE)/(W/deltat);

    /* ============================================== */
    if(filfac > FILFACCRIT){

    /* -- Determine H tilde */
    denA=4.0*L*L*float(nA-nC+nE)+W*W*float(nB+nD)+8.0*W*(tB-tD)+16.0*t2BD;
    numB=8.0*L*L*xAEC-4.0*W*L*xBD+16.0*L*(xtD-xtB);
    Hmax=-numB/denA;
    Hmaxh=Hmax/2.0;

    /* -- search the region A to E*/
    res=0.0;
    i=0;
    ele=0.0;

    for (int i=nSx; i<nSx+nAx; i++){
    if(cache[i+nsc] >= tStart){
    ele=cache[i] - mean +Hmaxh;
    res = res+ele*ele;
    }
    }

    for (int i=nSx+nAx; i<nSx+nAx+nBx; i++){
    if(cache[i+nsc] >= tStart){
    ele=cache[i]  - mean - Hmax/L*(cache[i+nsc]+W/4.0);
    res = res+ele*ele;
    }
    }

    for (int i=nSx+nAx+nBx; i<nSx+nAx+nBx+nCx; i++){
    if(cache[i+nsc] >= tStart){
    ele=cache[i] - mean -Hmaxh;
    res = res+ele*ele;
    }
    }

    for (int i=nSx+nAx+nBx+nCx; i<nSx+nAx+nBx+nCx+nDx; i++){
    if(cache[i+nsc] >= tStart){
    ele=cache[i] - mean + Hmax/L*(cache[i+nsc]-W/4.0);
    res = res+ele*ele;
    }
    }

    for (int i=nSx+nAx+nBx+nCx+nDx; i<nSx+nAx+nBx+nCx+nDx+nEx; i++){
    if(cache[i+nsc] >= tStart){
    ele=cache[i] - mean +Hmaxh;
    res = res+ele*ele;
    }
    }

    if(res>0.0){
    /* (S/N)**2 = height*height/(chisq/dof) */
    res=Hmax/sqrt(res/(nA+nB+nC+nD+nE-3)); 
    /* res=Hmax*Hmax/(res/(nA+nB+nC+nD+nE-3)); */

    if(res > res_max){
    res_max = res;
    Lmax = L;
    Hmaxmax = Hmax;

    }
    }

    /* the end of if filfac */ 
    }

    /* the end of the main loop */
    } 

    /* input maximum res_max into cache */
    cache[2*nsc+ithread]=res_max;

    __syncthreads();

    /* computing thread max */
    j = nthread/2;
    while (j !=0){
    if(ithread < j){
    cache[2*nsc+ithread] = max(cache[2*nsc+ithread],cache[2*nsc+ithread+j]);
    }
    __syncthreads();
    j /= 2;         
    }
    __syncthreads();

    /* input it to the global memory */ 
    if (ithread==0){
    tlssn[k] = cache[2*nsc];
    }

    if(res_max==cache[2*nsc]){
    tlst0[k]=deltat*float(blockIdx.x);
    tlsw[k]=W;  
    tlsl[k]=Lmax;  
    tlshmax[k]=Hmaxmax;  

    }


    




    }

    """,options=['-use_fast_math'])

    return source_module

if __name__ == "__main__":

    print ("gpu tls")
    import time
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.compiler
    from pycuda.compiler import SourceModule
    import sys
    import gtls_simple
    import gfilter
    import getstat
    import read_keplerlc as kep
    start = time.time()

    dirlist=["/sharksuck/kic/data/0038/003835482","/sharksuck/kic/data/0085/008510748"]
    lc,tu,n,ntrue,nq,inval, bjdoffset, tu0=kep.load_keplc(dirlist)

    print(tu0)
    print(bjdoffset)
    ## for transit ##
    lc = 2.0 - lc
    tsw=-1
    print("Transit Mode")
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
    wmax = 4.0  # day
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

    source_module=gtls_simple.gtls_module()
    
    ##compute kma,kmi,kkmi
    sharedsize=(2*nsc + nw + 2)*4 #byte
    print("sharedsize=",sharedsize)
    #gtls
    start = time.time()
    source_module=gtls_simple.gtls_module()
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

    iq=1
    fac=1.0
    i = np.argmax(tlssn[iq::2])
    im=np.max([0,int(i-nsc)])
    ix=np.min([nt,int(i+nsc)])
    
    llc=2.0 - lc[im:ix,iq]
    ttc=tu[im:ix,iq]

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.xlim(tlst0[iq::2][i]-tlsw[iq::2][i]+tu0,tlst0[iq::2][i]+tlsw[iq::2][i]+tu0)
    ax.plot(tu[im:ix,iq]+tu0,llc,".")
    mask=(ttc>0.0)
    plt.ylim(np.min(llc[mask]),np.max(llc[mask]))
    plt.ylabel("t")
    plt.xlabel("BKJD")

    plt.show()

