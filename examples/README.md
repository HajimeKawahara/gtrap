
- gtls_mockkepler.py -- generate mock template after the TLS identifier
 use genGroundTrues.sh for sequential use.
- astronet.py -- a KERAS version of astronet

- gtls_pickkepler.py pickup by TLS for all kepler LCs
- astronet_picklc.py prediction for all kepler LCs


## Tuning of TLS

test_pick_kelp.sh : test of TLS algorithm using the clean sample of the KeLP catalog.
test_pick_kelp.py : comparison

results:
-----------------------------------------------
for kepler KeLP Clean

gfilter smoothing 
128 54 / 61
100 60 / 67
64 55 / 63

the near edge problem was solved.

Next 100 60 / 67, 7 sample failed. why?

n=3 60/67
n=10 64/67
