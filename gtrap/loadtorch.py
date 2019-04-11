### torch packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn


class SteDataSet(Dataset):
    
    '''
    
    PURPOSE: DATA LOADER FOR KERPLER LIGHT CURVES
    INPUT: PATH TO DIRECTOR WITH LIGHT CURVES + INFO FILES
    OUTPUT: LOCAL + GLOBAL VIEWS, LABELS
    
    '''

    def __init__(self, filepath):

        ### list of global, local, and info files (assumes certain names of files)
        self.flist_global = np.sort(glob.glob(os.path.join(filepath, '*global.npy')))
        self.flist_local = np.sort(glob.glob(os.path.join(filepath, '*local.npy')))
        self.flist_info = np.sort(glob.glob(os.path.join(filepath, '*info.npy')))

        self.ids = np.sort([(x.split('/')[-1]).split('_')[1] + '_' + (x.split('/')[-1]).split('_')[2] for x in self.flist_global])


    def __len__(self):

        return self.ids.shape[0]

    def __getitem__(self, idx):

        ### grab local and global views
        data_global = np.load(self.flist_global[idx])
        data_local = np.load(self.flist_local[idx])

        ### grab centroid views
        data_global_cen = np.load(self.flist_global_cen[idx])
        data_local_cen = np.load(self.flist_local_cen[idx])
        
        ### info file contains: [0]kic, [1]tce, [2]period, [3]epoch, [4]duration, [5]label)
        data_info = np.load(self.flist_info[idx])
        
        return (data_local, data_global, data_local_cen, data_global_cen, data_info[6:]), data_info[5]


    
