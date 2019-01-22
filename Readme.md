gtrap: GPU transiting planet candidate detector

Trapezoid/Box least square (BLS) codes writtein in pycuda.

##gtls

GPU-based TLS.

- gtls_simple: simple gpu-based TLS
- gtls_kepler: for the Kepler data

##geebls

GPU-based BLS, pycuda-version of eebls.f (Kovac+2002). The codes use single precision float array. This requirement makes slight difference in the result between eebls.f and this code. Shared memory is used to cache folded light curve and temporary Signal Residue (SR). By default, GBLS allows a batch computation. An input array should be an image of light curves, which contains a set of N-light curves. 

- geebls_simple: simple gpu-based BLS
- geebls: gpu-based BLS with mask, offset, and a non-common time array, phase information

mask: Masked bins are ignored for the BLS computation. Values in a time array should be negative for the masked indices.
offset: This allows an extrenal input of the offset of the lightcurve.
non-common time array: This allows a different time sequence for each light curve. Set the time array which has the same dimension to that of the lightcurve image array.

Smoothing of light curves before the BLS generally increases the BLS signal. 

##scipy.signal.medfilt
scipy.signal.medfilt provides a good detrending method.

##gfilter
GBLS also has a gpu-based smoother, which devide a light curve by the median filtered curve.

- gfilter: a gpu-based smoother

The algorithm for the fast median filter is based on the chapter of "Implementing a fast median filter" by Gilles Perrot from the textbook, "Designing Scientific Applications on GPUs".

### best number of grids

~300,000

## Kepler LLC data

###Note on file format

The original fits data sometimes requires reading time comparable to the gbls itself. I recommend HDF format or feather format.
