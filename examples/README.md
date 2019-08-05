
# NN
How to use astronet...
/bin/bash
conda activate tf-gpu


# TESS (SLC)

- gtls_slctess.py -- mock and pick by the TLS

## When updating the LCs

### generate data list 



### generate training set for astronet

- gtls_slctess.py -q (no injection)
- gtls_slctess.py (injection)

### testing settings

- 

### pick up pulses by TLS

- gtls_slctess.py -p 

### do NN for picked-up data

-

# Kepler STE

- gtls_mockkepler.py -- generate mock template after the TLS identifier
 use genGroundTrues.sh for sequential use.
- astronet.py -- a KERAS version of astronet (training)
- gtls_pickkepler.py pickup by TLS for all kepler LCs
- astronet_picklc.py prediction for all kepler LCs


## Tuning of TLS

test_pick_kelp.sh : test of TLS algorithm using the clean sample of the KeLP catalog.
test_pick_kelp.py : comparison

