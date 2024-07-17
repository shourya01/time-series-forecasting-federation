# here we write some basic code which allows the creation of a dataloader

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Union, List, Tuple
from itertools import combinations
from torch.utils.data import random_split, Subset

def split_dataset(dataset, ratio, method='random'):
    # function to split a dataset according to ratio in (0,1)
    # method can be 'random' or 'sequential'
    train_size = int(len(dataset) * ratio)
    test_size = len(dataset) - train_size
    if method == 'random':
        return random_split(dataset, [train_size, test_size])
    elif method == 'sequential':
        return Subset(dataset, range(train_size)), Subset(dataset, range(train_size, len(dataset)))
    else:
        raise ValueError("Method must be 'random' or 'sequential'")

class TensorList(list):
    def __init__(self, items=[]):
        if not all(isinstance(item, (torch.Tensor, np.ndarray)) for item in items):
            raise ValueError("All items must be torch.Tensor or numpy.ndarray")
        super().__init__(items)

    def __getattr__(self, name):
        if len(self) == 0:
            raise AttributeError("TensorList is empty")
        if hasattr(torch.Tensor, name) or hasattr(np.ndarray, name):
            def method(*args, **kwargs):
                results = []
                for item in self:
                    attr = getattr(item, name, None)
                    if callable(attr):
                        result = attr(*args, **kwargs)
                    else:
                        result = attr
                    results.append(result)
                return TensorList(results)
            return method
        else:
            raise AttributeError(f"'TensorList' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.__dict__:
            # Set instance attribute
            super().__setattr__(name, value)
        else:
            # Set attribute on each item in the list
            for item in self:
                setattr(item, name, value)

    def append(self, item):
        if not isinstance(item, (torch.Tensor, np.ndarray)):
            raise TypeError("Item must be a torch.Tensor or numpy.ndarray")
        super().append(item)

    def to(self, *args, **kwargs):
        # Custom method to handle torch.Tensor's `to` method, which is not present in numpy.ndarray
        return TensorList([item.to(*args, **kwargs) if isinstance(item, torch.Tensor) else item for item in self])
        
class LFDataset(Dataset):
    
    def __init__(
        self,
        data_y_s: np.ndarray,
        data_x_u: np.ndarray,
        lookback: int,
        lookahead: int,
        client_idx: int,
        idx_x: Union[List,Tuple],
        idx_u: Union[List,Tuple],
        dtype: torch.dtype = torch.float32
    ):
        
        # sanity checks
        assert lookback > 0, "Cannot have non-positive lookback!"
        assert lookahead > 0, "Cannot have non-positive lookahead!"
        assert client_idx < data_y_s['load'].shape[0], "Client index exceeds number of clients present."
        assert len(idx_x)+len(idx_u) == data_x_u['wdata'].shape[0], "Indices provided do not sum upto the input dimension."
        assert all(not set(a) & set(b) for a, b in combinations([idx_x, idx_u], 2)), "All indices are not mutually exclusive."
        
        # save inputs
        self.load = data_y_s['load'][client_idx,:]
        self.static = data_y_s['static'][client_idx,:]
        self.x, self.u = data_x_u['wdata'][idx_x,:], data_x_u['wdata'][idx_u,:]
        self.idx_x, idx_u = idx_x, idx_u
        self.lookback, self.lookahead = lookback, lookahead
        self.dtype = dtype
        
        # max length
        self.maxlen = self.load.shape[0] - lookback - lookahead + 1
        
    def __len__(self):
        
        return self.maxlen
    
    def __getitem__(self, idx):
        
        y_past = torch.tensor(self.load[idx:idx+self.lookback][:,None], dtype=self.dtype)
        x_past = torch.tensor(self.x[:,idx:idx+self.lookback].T, dtype=self.dtype)
        u_past = torch.tensor(self.u[:,idx:idx+self.lookback].T, dtype=self.dtype)
        u_future = torch.tensor(self.u[:,idx+self.lookback:idx+self.lookback+self.lookahead].T, dtype=self.dtype)
        s_past = torch.tensor(self.static[None,:].repeat(self.lookback,axis=0), dtype=self.dtype)
        y_target = torch.tensor(self.load[idx+self.lookback+self.lookahead-1].reshape((1,)), dtype=self.dtype)
        y_all_target = torch.tensor(self.load[idx+self.lookback:idx+self.lookback+self.lookahead][:,None], dtype=self.dtype)
        
        inp = TensorList((y_past,x_past,u_past,s_past,u_future,y_all_target)) # last one is for teacher forcing during training
        lab = TensorList((y_target, y_all_target))
        
        return inp, lab