# here we write some basic code which allows the creation of a dataloader

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Union, List, Tuple
from itertools import combinations
from torch.utils.data import random_split, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List

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

def custom_scaler(mean, var, len_features, num):
    # custom scaler, used later to create fixed point scalers
    scaler = StandardScaler()
    scaler.mean_ = mean*np.ones(len_features)
    scaler.var_ = var*np.ones(len_features)
    scaler.scale_ = np.sqrt(var)*np.ones(len_features)
    scaler.n_samples_seen = num
    return scaler

def combine_standard_scalers(scalers):
    # combine multiple StandardScalers into one
    total_samples = sum(scaler.n_samples_seen_ for scaler in scalers)
    combined_mean = sum(scaler.mean_ * scaler.n_samples_seen_ for scaler in scalers) / total_samples
    combined_variance = (
        sum(scaler.var_ * scaler.n_samples_seen_ + scaler.mean_**2 * scaler.n_samples_seen_ for scaler in scalers) / total_samples
        - combined_mean**2
    )
    combined_scaler = StandardScaler()
    combined_scaler.mean_ = combined_mean
    combined_scaler.var_ = combined_variance
    combined_scaler.scale_ = np.sqrt(combined_variance)
    combined_scaler.n_samples_seen_ = total_samples
    return combined_scaler

def combine_minmax_scalers(scalers):
    # combine multiple MinMaxScalers into one
    combined_data_min = np.min([scaler.data_min_ for scaler in scalers], axis=0)
    combined_data_max = np.max([scaler.data_max_ for scaler in scalers], axis=0)
    combined_scaler = MinMaxScaler()
    combined_scaler.data_min_ = combined_data_min
    combined_scaler.data_max_ = combined_data_max
    combined_scaler.data_range_ = combined_data_max - combined_data_min
    combined_scaler.feature_range = scalers[0].feature_range  # Assuming all scalers have the same feature range
    return combined_scaler
        
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
        normalize_type: str = None, # minmax, z or None
        ratio: float = 0.8
    ):
        
        # sanity checks
        assert lookback > 0, "Cannot have non-positive lookback!"
        assert lookahead > 0, "Cannot have non-positive lookahead!"
        assert client_idx < data_y_s['load'].shape[0], "Client index exceeds number of clients present."
        assert len(idx_x)+len(idx_u) == data_x_u['wdata'].shape[0], "Indices provided do not sum upto the input dimension."
        assert all(not set(a) & set(b) for a, b in combinations([idx_x, idx_u], 2)), "All indices are not mutually exclusive."
        
        # save inputs
        self.y = data_y_s['load'][client_idx,:][None,:].T
        self.x, self.u = data_x_u['wdata'][idx_x,:].T, data_x_u['wdata'][idx_u,:].T
        self.s = data_y_s['static'][client_idx,:][:,None].repeat(self.y.size,axis=1).T
        
        # save lookbacks and data types
        self.lookback, self.lookahead = lookback, lookahead
        self.dtype = dtype
        
        # max length
        self.maxlen = self.y.shape[0] - lookback - lookahead + 1
        
        # carry out mean-std normalization
        self.normalize_type = normalize_type
        self.ttr = ratio
        self._generate_stats()
        
    def __len__(self):
        
        return self.maxlen
    
    def __getitem__(self, idx):
        
        y_past, x_past, u_past, s_past = [
            itm[idx:idx+self.lookback,:] for itm in [self.y, self.x, self.u, self.s]
        ]
        
        y_future, u_future = [
            itm[idx+self.lookback:idx+self.lookback+self.lookahead,:] for itm in [self.y, self.u]
        ]
        
        # normalize
        y_future_copy = y_future.copy()
        if self.normalize_type == 'minmax' or self.normalize_type == 'z':
            y_past, y_future, x_past, u_past, u_future, s_past = self._transform(y_past, y_future, x_past, u_past, u_future, s_past)
            if self.normalize_type == 'minmax':
                fac_0, fac_1 = self.y_scaler.data_min_.item(), self.y_scaler.data_max_.item()
            else:
                fac_0, fac_1 = self.y_scaler.mean_.item(), self.y_scaler.scale_.item()
        else:
            fac_0, fac_1 = 0., 1.
            
        # convert to tensors
        y_past, y_future, x_past, u_past, u_future, s_past = [
            torch.tensor(itm,dtype=self.dtype) for itm in [y_past, y_future, x_past, u_past, u_future, s_past]
        ]
        y_future_copy = torch.tensor(y_future_copy, dtype=self.dtype)

        # stuff factors into output
        stuffed_fac0 = fac_0 * torch.ones_like(y_future, dtype=self.dtype)
        stuffed_fac1 = fac_1 * torch.ones_like(y_future, dtype=self.dtype)
        
        past_cat = torch.cat([y_past,x_past,u_past,s_past],dim=-1)
        fut_cat = torch.cat([y_future,u_future],dim=-1) # while we include y_future, it is only for teacher forcing. In a real implementation it can be replaced with zeros, since it is not used anyways when self.training is False
        
        inp = pad_and_concatenate(past_cat,fut_cat)
        lab = torch.cat((y_future_copy,stuffed_fac0,stuffed_fac1),dim=-1)
        
        return inp, lab
    
    def _generate_stats(self):
        
        if self.normalize_type == 'minmax':
            self.y_scaler, self.x_scaler, self.u_scaler, self.s_scaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
        else:
            if self.normalize_type == 'z':
                self.y_scaler, self.x_scaler, self.u_scaler, self.s_scaler = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
            else:
                self.y_scaler = custom_scaler(0., 1., self.y.shape[1], self.y.shape[0])
                self.x_scaler = custom_scaler(0., 1., self.x.shape[1], self.x.shape[0])
                self.u_scaler = custom_scaler(0., 1., self.u.shape[1], self.u.shape[0])
                self.s_scaler = custom_scaler(0., 1., self.s.shape[1], self.s.shape[0])
                
        if self.normalize_type == 'minmax' or self.normalize_type == 'z':
            self.y_scaler, self.x_scaler, self.u_scaler, self.s_scaler = self.y_scaler.fit(self.y[:int(self.y.shape[0]*self.ttr),:]), self.x_scaler.fit(self.x[:int(self.x.shape[0]*self.ttr),:]), self.u_scaler.fit(self.u[:int(self.u.shape[0]*self.ttr),:]), self.s_scaler.fit(self.s[:int(self.s.shape[0]*self.ttr),:])
            
    def _get_scalers(self):
        
        return self.y_scaler, self.x_scaler, self.u_scaler, self.s_scaler
    
    def _combine_scalers(
        self,
        y_scalers: List,
        x_scalers: List,
        u_scalers: List,
        s_scalers: List
    ):
        
        if self.normalize_type == 'minmax':
            self.y_scaler = combine_minmax_scalers(y_scalers)
            self.x_scaler = combine_minmax_scalers(x_scalers)
            self.u_scaler = combine_minmax_scalers(u_scalers)
            self.s_scaler = combine_minmax_scalers(s_scalers)
        else:
            if self.normalize_type == 'z':
                self.y_scaler = combine_standard_scalers(y_scalers)
                self.x_scaler = combine_standard_scalers(x_scalers)
                self.u_scaler = combine_standard_scalers(u_scalers)
                self.s_scaler = combine_standard_scalers(s_scalers) 
        
                          
    def _transform(self, y_past, y_future, x, u_past, u_future, s):
        
        return self.y_scaler.transform(y_past), self.y_scaler.transform(y_future), self.x_scaler.transform(x), self.u_scaler.transform(u_past), self.u_scaler.transform(u_future), self.s_scaler.transform(s)
    
    def _get_full_normalized_data(self):
        
        ynorm = self.y_scaler.transform(self.y)
        xnorm = self.x_scaler.transform(self.x)
        unorm = self.u_scaler.transform(self.u)
        snorm = self.s_scaler.transform(self.s)
        return ynorm, xnorm, unorm, snorm
    
    def _get_full_unnormalized_data(self):
        
        return self.y, self.x, self.u, self.s