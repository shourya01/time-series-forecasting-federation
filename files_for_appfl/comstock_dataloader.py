# This dataloader assumes two things exist: 
# - the COMSTOCK building data separated by district and stored in .npz files 
# (argonne users with access, find the same at "/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/grouped/")
# - a text file containing multiple lines, each line having data <districtID,num_buildings>
# (argonne users with access, find the same at "/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/client_count.txt")

import sys, os
import numpy as np
from typing import Union, List, Tuple
import torch
import itertools
import importlib

# user-specific, see notice above
# FUNCTION: Import function
def import_function(file_path, function_name):
    # Derive module name from file path
    module_name = file_path.replace('/', '.').rstrip('.py')
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Access the function
    return getattr(module, function_name)
LFDataset = import_function('/home/sbose/time-series-forecasting-federation/models/LFDataset.py','LFDataset')
split_dataset = import_function('/home/sbose/time-series-forecasting-federation/models/LFDataset.py','split_dataset')

DEFAULT_FNAME = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/client_count.txt'
DEFAULT_DATA_DIR = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/grouped'

def get_bldg_idx(
    bldg_idx: int,
    bldg_counts: Union[List,Tuple]
    ):
    '''
    inputs: 
    bldg_idx: overall index of desired building
    bldg_counts: list or tuple containing building counts for each district
    output:
    bldg_in_district_idx: index of desired building in its district
    district_idx: district index
    '''
    if bldg_idx < 0 or bldg_idx >= sum(bldg_counts):
        raise ValueError("Building index out of bounds of dataset.")
    district_idx =  next(i for i, x in enumerate(itertools.accumulate(bldg_counts)) if x > bldg_idx)
    bldg_in_district_idx = bldg_idx - sum(bldg_counts[:district_idx])
    return bldg_in_district_idx, district_idx

def get_comstock(
    bldg_idx: int,
    lookback: int = 12,
    lookahead: int = 4,
    idx_x: Union[List,Tuple] = [0,1,2,3,4,5],
    idx_u: Union[List,Tuple] = [6,7],
    train_test_ratio: float = 0.8,
    bldg_list_file: str = DEFAULT_FNAME, # change upon different usage
    bldg_data_dir: str = DEFAULT_DATA_DIR, # change upon different usage
    normalize: str = False, # normalize data
    normalize_type: str = 'minmax',
    dtype: torch.dtype = torch.float32,
    ):
    
    # load the list of files that contain our data
    datafiles = []
    with open(bldg_list_file,'r') as file:
        for line in file:
            string, number = line.strip().split(',')
            datafiles.append((string,int(number)))
            
    # get indices
    bldg_counts = [b for _,b in datafiles]
    bldg_in_district_idx, district_idx = get_bldg_idx(bldg_idx,bldg_counts)
    
    # load data
    data_y_s = np.load(bldg_data_dir+f'/{datafiles[district_idx][0]}_data.npz')
    data_x_u = np.load(bldg_data_dir+f'/{datafiles[district_idx][0]}_weather.npz')
    
    # create dataset
    dset = LFDataset(
        data_y_s = data_y_s,
        data_x_u = data_x_u,
        lookback = lookback,
        lookahead = lookahead,
        client_idx = bldg_in_district_idx,
        idx_x = idx_x,
        idx_u = idx_u,
        dtype = dtype,
        normalize = normalize,
        normalize_type = normalize_type,
        ratio = train_test_ratio
    )
    
    # split into train and test sets
    d_train, d_test = split_dataset(dset, train_test_ratio, method='sequential')
        
    return d_train, d_test