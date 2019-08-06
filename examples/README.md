
# NN
How to use astronet...
/bin/bash
conda activate tf-gpu


# TESS (SLC)

- gtls_slctess.py -- mock and pick by the TLS

## When updating the LCs

- 0.1 generate data list 

data/python_updatelist/make_list_sector.py
list are in data/ctl.list

- 1. generate training datasets

```
gtls_slctess.py -q (no injection)
gtls_slctess.py (injection)
```

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

