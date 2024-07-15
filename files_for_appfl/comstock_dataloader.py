# This dataloader assumes two things exist: 
# - the COMSTOCK building data separated by district and stored in .npz files 
# (argonne users with access, find the same at "/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/grouped/")
# - a text file containing multiple lines, each line having data <districtID,num_buildings>
# (argonne users with access, find the same at "/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/client_count.txt")

import os, sys
import numpy as np
from typing import Union, List, Tuple
import torch

# user-specific, see notice above
sys.path.insert(0,'/home/sbose/time-series-forecasting-federation')
from models.LFDataset import LFDataset, split_dataset

DEFAULT_FNAME = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/client_count.txt'
DEFAULT_DATA_DIR = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/grouped'

def get_comstock(
    
    num_bldg: int,
    lookback: int = 8,
    lookahead: int = 4,
    idx_x: Union[List,Tuple] = [0,1,2,3,4,5],
    idx_u: Union[List,Tuple] = [6,7],
    train_test_ratio: float = 0.8,
    bldg_list_file: str = DEFAULT_FNAME,
    bldg_data_dir: str = DEFAULT_DATA_DIR,
    dtype: torch.dtype = torch.float32
    ):
    
    # empty lists to store datasets
    train_dsets, test_dsets = [], []
    
    # load the list of files that contain our data
    datafiles = []
    with open(bldg_list_file,'r') as file:
        for line in file:
            string, number = line.strip().split(',')
            datafiles.append((string,int(number)))
            
    
    cur_district_idx, cur_district_building_idx = 0,0
    
    # loop across all clients
    for _ in range(num_bldg):
        
        # check if we have exhausted buildings for the current district, otherwise load npz file and move the index ahead
        if cur_district_building_idx == datafiles[cur_district_idx][1]:
            cur_district_idx += 1
            cur_district_building_idx = 0
        data_y_s = np.load(bldg_data_dir+f'/{datafiles[cur_district_idx][0]}_data.npz')
        data_x_u = np.load(bldg_data_dir+f'/{datafiles[cur_district_idx][0]}_weather.npz')
        
        dset = LFDataset(
            data_y_s = data_y_s,
            data_x_u = data_x_u,
            lookback = lookback,
            lookahead = lookahead,
            client_idx = cur_district_building_idx,
            idx_x = idx_x,
            idx_u = idx_u,
            dtype = dtype
        )
        
        d_train, d_test = split_dataset(dset, train_test_ratio, method='sequential')
        train_dsets.append(d_train)
        test_dsets.append(d_test)
        
    return train_dsets, test_dsets

    
    
    
    