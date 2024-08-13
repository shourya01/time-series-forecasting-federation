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
    
def pad_and_concatenate(tensor1, tensor2):
    # take two 2D tensors, measure the first dims of each,
    # and pad with zeros to make the first dim same-sized.
    # then concatenate the resulting vector along the second
    # dimension
    dim1, sz1 = tensor1.shape
    dim2, sz2 = tensor2.shape
    if dim1 > dim2:
        padding = (0, 0, 0, dim1 - dim2)
        tensor2 = torch.nn.functional.pad(tensor2, padding)
    elif dim1 < dim2:
        print(f"Detected that dim1={dim1} is lesser than dim2={dim2}. Shape before padding: {tensor2.shape}.")
        tensor1 = torch.nn.functional.pad(tensor1, padding)
    return torch.cat((tensor1, tensor2), dim=1)
        
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
        dtype: torch.dtype = torch.float32,
        normalize: bool = True,
        ratio: float = 0.8
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
        
        # carry out mean-std normalization
        self.normalize = normalize
        self.ttr = ratio
        self._normalize()
        
        print(f"Normalize:{normalize}")
        
    def __len__(self):
        
        return self.maxlen
    
    def __getitem__(self, idx):
        
        if self.normalize:
            self.mu_y, self.std_y = self.y_factor_mean, self.y_factor_std
            self.mu_x, self.std_x = self.x_factor_mean, self.x_factor_std
            self.mu_u, self.std_u = self.u_factor_mean, self.u_factor_std
            self.mu_s, self.std_s = self.static_mean, self.static_std
        else:
            self.mu_y, self.std_y = 0., 1.
            self.mu_x, self.std_x = np.zeros_like(self.x_factor_mean), np.ones_like(self.x_factor_std)
            self.mu_u, self.std_u = np.zeros_like(self.u_factor_mean), np.ones_like(self.u_factor_std)
            self.mu_s, self.std_s = 0., 1.
        
        y_past = torch.tensor(self._transform(self.load[idx:idx+self.lookback][:,None],self.mu_y,self.std_y), dtype=self.dtype)
        x_past = torch.tensor(self._transform(self.x[:,idx:idx+self.lookback].T,self.mu_x.T,self.std_x.T), dtype=self.dtype)
        u_past = torch.tensor(self._transform(self.u[:,idx:idx+self.lookback].T,self.mu_u.T,self.std_u.T), dtype=self.dtype)
        u_future = torch.tensor(self._transform(self.u[:,idx+self.lookback:idx+self.lookback+self.lookahead].T,self.mu_u.T,self.std_u.T), dtype=self.dtype)
        s_past = torch.tensor(self._transform(self.static[None,:].repeat(self.lookback,axis=0),self.mu_s,self.std_s), dtype=self.dtype)
        y_all_target = torch.tensor(self._transform(self.load[idx+self.lookback:idx+self.lookback+self.lookahead][:,None],self.mu_y,self.std_y), dtype=self.dtype)
        
        past_cat = torch.cat([y_past,x_past,u_past,s_past],dim=-1)
        fut_cat = torch.cat([y_all_target,u_future],dim=-1) # while we include y_all_target, it is only for teacher forcing. In a real implementation it can be replaced with zeros, since it is not used anyways when self.training is False
        
        inp = pad_and_concatenate(past_cat,fut_cat)
        lab = y_all_target
        
        return inp, lab
    
    def _transform(self, x, mu, std):
        
        return (x-mu) / std
    
    def _normalize(self):
        
        # carry out mean-std normalization and save the factors
        
        self.y_factor_mean, self.y_factor_std = np.mean(self.load[:int(self.ttr*self.load.size)]), np.std(self.load[:int(self.ttr*self.load.size)])
        self.x_factor_mean, self.x_factor_std = np.mean(self.x[:,:int(self.ttr*self.x.shape[1])], axis=1, keepdims=True), np.std(self.x[:,:int(self.ttr*self.x.shape[1])], axis=1, keepdims=True)
        self.u_factor_mean, self.u_factor_std = np.mean(self.u[:,:int(self.ttr*self.u.shape[1])], axis=1, keepdims=True), np.std(self.u[:,:int(self.ttr*self.u.shape[1])], axis=1, keepdims=True)
        self.static_mean, self.static_std = np.mean(self.static), np.std(self.static)