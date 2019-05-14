import pandas as pd
import h5py
import numpy as np
dat=pd.read_csv("../data/step3.list")
np.savez("step3.list.npz",dat["FILE"])
