# process split data

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
from mpi4py import MPI

def split_dict(input_dict, n):
    dict_len = len(input_dict)
    split_size = dict_len // n
    it = iter(input_dict)
    return [{k: input_dict[k] for k in islice(it, split_size)} for _ in range(n - 1)] + [{k: input_dict[k] for k in it}]

base_dir = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA' # data directory
versionID = 32 # NREL Comstock upgrade version


if __name__ == "__main__":
    
    # MPI processes
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # read default PARQUET file
    metadata = pd.read_parquet(base_dir+f'/upgrade{versionID}.parquet')

    # extract relevant info as lists
    bldg_id_list = metadata.index.tolist() # building id List
    state_name_list = metadata['in.state'].tolist() # state name list
    county_id_list = metadata['in.nhgis_county_gisjoin'].tolist() # county gisjoin list
    floor_area_list = metadata['calc.weighted.sqft'].tolist() # floor area list
    wall_area_list = metadata['out.params.ext_wall_area..m2'].tolist() # wall area list
    window_area_list = metadata['out.params.ext_window_area..m2'].tolist() # window area list

    # group by county id
    grouped = defaultdict(list)
    for a,b,c,d,e,f in zip(bldg_id_list,state_name_list,county_id_list,floor_area_list,wall_area_list,window_area_list):
        grouped[c].append((a,b,c,d,e,f))
        
    # make directory to store datas
    os.makedirs(base_dir+'/grouped',exist_ok=True)
    
    # split grouped according to world size
    grouped_splits = split_dict(grouped,world_size)
        
    # loop across dict keys
    my_grouped = grouped_splits[mpi_rank]
    for key in tqdm(my_grouped.keys()):
        # read weather data
        weather_path = base_dir + f'/weather/{key}_2018.csv'
        if not os.path.exists(weather_path):
            continue
        else:
            weather_df = pd.read_csv(weather_path)
            db_temp = np.repeat(weather_df['Dry Bulb Temperature [°C]'].to_numpy(),4).tolist()
            w_speed = np.repeat(weather_df['Wind Speed [m/s]'].to_numpy(),4).tolist()
            dtidx_recorded = False # we will record daytime indices when we load the consumption files
            data1, data2, bid_recorder = [], [], []
            for a,b,c,d,e,f in grouped[key]:
                bldg_path = base_dir + f'/{b}/{a}-{versionID}.parquet'
                if os.path.exists(bldg_path):
                    try:
                        bldg_df = pd.read_parquet(bldg_path)
                    except:
                        print(f"Encountered error with reading parquet {bldg_path}, continuing.")
                        continue
                else:
                    print(f"{bldg_path} does not exist, continuing.")
                    continue
                consumption = bldg_df['out.electricity.total.energy_consumption'].tolist()
                if not dtidx_recorded:
                    time_idx = ((bldg_df['timestamp'] - bldg_df['timestamp'].dt.normalize()) / np.timedelta64(15, 'm')).astype(int).tolist()
                    day_idx = bldg_df['timestamp'].dt.dayofweek.tolist()
                    dtidx_recorded = True
                data1.append(consumption)
                data2.append([d,e,f])
                bid_recorder.append([a])
            if dtidx_recorded == False:
                continue
            wdata = [db_temp,w_speed,time_idx,day_idx]
            data1, data2, wdata = np.array(data1), np.array(data2), np.array(wdata)
            bid_data = np.array(bid_recorder)
            # save files
            np.savez_compressed(base_dir+f'/grouped/{key}_data.npz',load=data1,static=data2,bid=bid_data)
            np.savez_compressed(base_dir+f'/grouped/{key}_weather.npz',wdata=wdata)
            print(f"Processed {key} on process {mpi_rank+1}/{world_size}.",flush=True)
        