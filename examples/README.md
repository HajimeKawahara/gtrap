
# NN
How to use astronet...
/bin/bash
conda activate tf-gpu


# TESS (SLC)

## When updating the LCs

- 0.1 generate data list 




- gtls_slctess.py -- mock and pick by the TLS


# Kepler STE

- gtls_mockkepler.py -- generate mock template after the TLS identifier
 use genGroundTrues.sh for sequential use.
- astronet.py -- a KERAS version of astronet (training)
- gtls_pickkepler.py pickup by TLS for all kepler LCs
- astronet_picklc.py prediction for all kepler LCs


## Tuning of TLS

test_pick_kelp.sh : test of TLS algorithm using the clean sample of the KeLP catalog.
test_pick_kelp.py : comparison

