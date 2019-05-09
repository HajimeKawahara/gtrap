### torch packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


class SteDataSet(Dataset):
    
    '''
    
    PURPOSE: DATA LOADER FOR KERPLER LIGHT CURVES
    INPUT: PATH TO DIRECTOR WITH LIGHT CURVES + INFO FILES
    OUTPUT: LOCAL + GLOBAL VIEWS, LABELS
    
    '''

    def __init__(self, filepath):

        ### list of global, local, and info files (assumes certain names of files)
        self.flist = np.sort(glob.glob(os.path.join(filepath, 'mocklc_clean/mock*.npz')))


    def __len__(self):

        return self.ids.shape[0]

    def __getitem__(self, idx):

        ### grab local and global views
        d = np.load(self.flist[idx])
        
        ### info file contains: [0]kic, [1]tce, [2]period, [3]epoch, [4]duration, [5]label)
        
        return (d["arr_1"], d["arr_2"]), d["arr_0"]


if __name__ == "__main__":
    d = np.load("../examples/mocklc_clean/mock5252588.npz")
    print(d["arr_0"]) #label
    print(np.shape(d["arr_1"])) #local
    print(np.shape(d["arr_2"])) #global
