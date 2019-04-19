import batman
import astropy.units as u
from astropy.constants import G, R_sun, M_sun, R_jup, M_jup
import numpy as np
import matplotlib.pyplot as plt
import picktrapcand as ptc
import read_keplerlc as kep
import argparse

def gentransit(t,t0=0.0,Porb=600.0,Rp=1.0,Mp=1.0,Rs=1.0,Ms=1.0,ideg=90.0,w=90.0,e=0.0,u1=0.1,u2=0.3):
    #mock LC
    #Rp [Rj]
    params = batman.TransitParams() 
    params.t0 = t0 # time of inferior conjunction 
    params.rp = Rp*R_jup/(Rs*R_sun) # planet radius (in units of stellar radii)
    
    # calculate semi-major axis from orbital period value
    params.inc = ideg  # orbital inclination (in degrees)
    params.ecc = e # eccentricity
    params.w = w # longitude of periastron (in degrees), 90 for circular
    params.u = [u1,u2] # limb darkening coefficients
    params.limb_dark = "quadratic" # limb darkening model

#   t = np.linspace(0.0, Porb*2, 72000) # times at which to calculate the light curve

    #period update
    params.per = Porb # orbital period (days)
    a = (((params.per*u.day)**2 * G * (Ms*M_sun + Mp*M_jup) / (4*np.pi**2))**(1./3)).to(R_sun).value/Rs     
    params.a = a # semi-major axis (in units of stellar radii)
    b=a*np.cos(ideg/180.0*np.pi)
    
    m = batman.TransitModel(params, t) # initializes the model
    injlc = np.array(m.light_curve(params))

    return injlc, b

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transit signal injection')
        
    parser.add_argument('-k', nargs=1, default=[1717722], help='kic', type=int)
    args = parser.parse_args()
    kicint=args.k[0]
    lctag="data"
    mydir="/sharksuck/kic/";
    kicdir=ptc.getkicdir(kicint,mydir+"data/")+"/"

    lc,tu,n,ntrue,nq,inval,bjdoffset,t0, t,det=kep.load_keplc([kicdir],offt="0")
    
    ilc,b=gentransit(t,t0=0.0,ideg=89.9,Porb=700.0)
    plt.plot(t,ilc*det,".")
    plt.show()
    
    
