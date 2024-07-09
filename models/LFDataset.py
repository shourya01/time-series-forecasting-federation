# here we write some basic code which allows the creation of a dataloader

import torch
from torch.utils.data import Dataset
from typing import Union, List, Tuple
from itertools import combinations

class LFDataset(Dataset):
    
    def __init__(
        self,
        data: np.array,
        lookback: int,
        lookahead: int,
        idx_y: Union[List,Tuple],
        idx_x: Union[List,Tuple],
        idx_u: Union[List,Tuple],
        idx_s: Union[List,Tuple],
        dtype: torch.dtype = torch.float32
    ):
        
        # sanity checks
        assert len(data.shape) == 2, "Incorrect number of dimensions in data."
        assert len(idx_y) > 0, "Cannot forecast indices of size 0"
        assert lookback > 0, "Cannot have non-positive lookback!"
        assert lookahead > 0, "Cannot have non-positive lookahead!"
        assert len(idx_y)+len(idx_x)+len(idx_u)+len(idx_s) == data.shape[1], "Indices provided do not sum upto the input dimension."
        assert all(not set(a) & set(b) for a, b in combinations([idx_y, idx_x, idx_u, idx_s], 2)), "All indices are not mutually exclusive."
        assert data.shape[0] >= lookback+lookahead, "Data too short to generate even 1 sample!"
        
        # save inputs
        self.data, self.dtype = data, dtype
        self.lookback, self.lookahead = lookback, lookahead
        self.idx_y, self.idx_x, self.idx_u, self.idx_s = idx_y, idx_x, idx_u, idx_s
        
        # generate datas
        self.records = []
        for tidx in range(self.data.shape[0]-self.lookback-self.lookahead+1):
            y_past = torch.tensor(self.data[tidx:tidx+lookback,idx_y], dtype=self.dtype)
            x_past = torch.tensor(self.data[tidx:tidx+lookback,idx_x], dtype=self.dtype)
            u_past = torch.tensor(self.data[tidx:tidx+lookback,idx_u], dtype=self.dtype)
            s_past = torch.tensor(self.data[tidx:tidx+lookback,idx_s], dtype=self.dtype)
            y_target = torch.tensor(self.data[tidx+lookback+lookahead-1,idx_y], dtype=self.dtype)
            y_all_target = torch.tensor(self.data[tidx+lookback:tidx+lookback+lookahead,idx_y], dtype = dtype)
            u_future = torch.tensor(self.data[tidx:tidx+lookback:tidx+lookback+lookahead,idx_u], dtype=self.dtype)
            self.records.append((y_past,x_past,u_past,s_past,y_target,y_all_target,u_future))
        
    def __len__(self):
        
        return len(self.records)
    
    def __getitem__(self, idx):
        
        record = self.records[idx]
        y_past, x_past, u_past, s_past, y_target, y_all_target, u_future = record
        inp = (y_past,x_past,u_past,s_past,u_future)
        lab = (y_target, y_all_target)
        
        return inp, lab