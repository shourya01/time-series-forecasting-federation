# process split data
# rewriting of process.py using concurrent.futures instead of mpi

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def split_dict(input_dict, n):
    dict_len = len(input_dict)
    split_size = dict_len // n
    it = iter(input_dict)
    return [{k: input_dict[k] for k in islice(it, split_size)} for _ in range(n - 1)] + [{k: input_dict[k] for k in it}]

base_dir = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA' # data directory
versionID = 32 # NREL Comstock upgrade version
num_cores = os.cpu_count() # Number of cpu cores available for the job

# read default PARQUET file
metadata = pd.read_parquet(base_dir+f'/upgrade{versionID}.parquet')

# extract relevant info as lists
bldg_id_list = metadata.index.tolist() # building id List
state_name_list = metadata['in.state'].tolist() # state name list
county_id_list = metadata['in.nhgis_county_gisjoin'].tolist() # county gisjoin list
floor_area_list = metadata['in.sqft'].tolist() # floor area list
wall_area_list = metadata['out.params.ext_wall_area..m2'].tolist() # wall area list
window_area_list = metadata['out.params.ext_window_area..m2'].tolist() # window area list
# adding more static features
nspaces_list = metadata['out.params.number_of_spaces'].tolist() # number of spaces list
nzones_list = metadata['out.params.number_of_zones'].tolist() # number os zones list
nsurfaces_list = metadata['out.params.number_of_surfaces'].tolist() # number of surfaces list
cooling_cap_list = metadata['out.params.cooling_equipment_capacity..tons'].tolist() # cooling equipment capacity in tons
# group by county id
grouped = defaultdict(list)
for a,b,c,d,e,f,g1,g2,g3,g4 in zip(bldg_id_list,state_name_list,county_id_list,floor_area_list,wall_area_list,window_area_list,nspaces_list,nzones_list,nsurfaces_list,cooling_cap_list):
    grouped[c].append((a,b,c,d,e,f,g1,g2,g3,g4))

# split grouped according to world size
grouped_splits = split_dict(grouped,num_cores)

# parallel function to execute
def parallel_function(my_grouped):
    
    total_len = len(my_grouped.keys())
    for kidx,key in enumerate(my_grouped.keys()):
        # read weather data
        weather_path = base_dir + f'/weather/{key}_2018.csv'
        if not os.path.exists(weather_path):
            print(f"Weather for district {key} not found. Continuing.",flush=True)
            continue
        else:
            weather_df = pd.read_csv(weather_path)
            db_temp = np.repeat(weather_df['Dry Bulb Temperature [°C]'].to_numpy(),4).tolist()
            w_speed = np.repeat(weather_df['Wind Speed [m/s]'].to_numpy(),4).tolist()
            wf1 = np.repeat(weather_df['Global Horizontal Radiation [W/m2]'].to_numpy(),4).tolist() # new weather feature 1
            wf2 = np.repeat(weather_df['Diffuse Horizontal Radiation [W/m2]'].to_numpy(),4).tolist() # new weather feature 2
            wf3= np.repeat(weather_df['Direct Normal Radiation [W/m2]'].to_numpy(),4).tolist() # new weather feature 3
            wf4 = np.repeat(weather_df['Wind Direction [Deg]'].to_numpy(),4).tolist() # new weather feature 4
            dtidx_recorded = False # we will record daytime indices when we load the consumption files
            data1, data2, bid_recorder = [], [], []
            for a,b,c,d,e,f,g1,g2,g3,g4 in my_grouped[key]:
                bldg_path = base_dir + f'/{b}/{a}-{versionID}.parquet'
                if os.path.exists(bldg_path):
                    try:
                        bldg_df = pd.read_parquet(bldg_path)
                    except:
                        print(f"Encountered error with reading parquet {bldg_path}, continuing.",flush=True)
                        continue
                else:
                    print(f"{bldg_path} does not exist, continuing.",flush=True)
                    continue
                consumption = bldg_df['out.electricity.total.energy_consumption'].tolist()
                if not dtidx_recorded:
                    time_idx = ((bldg_df['timestamp'] - bldg_df['timestamp'].dt.normalize()) / np.timedelta64(15, 'm')).astype(int).tolist()
                    day_idx = bldg_df['timestamp'].dt.dayofweek.tolist()
                    dtidx_recorded = True
                data1.append(consumption)
                data2.append([d,e,f,g1,g2,g3,g4])
                bid_recorder.append([a])
            if dtidx_recorded == False:
                continue
            wdata = [db_temp,w_speed,wf1,wf2,wf3,wf4,time_idx,day_idx]
            data1, data2, wdata = np.array(data1), np.array(data2), np.array(wdata)
            bid_data = np.array(bid_recorder)
            # save files
            np.savez_compressed(base_dir+f'/grouped/{key}_data.npz',load=data1,static=data2,bid=bid_data)
            np.savez_compressed(base_dir+f'/grouped/{key}_weather.npz',wdata=wdata)
            print(f"Processed {key} on process {os.getpid()}. Done {kidx+1}/{total_len}",flush=True)
            
    return 0


if __name__ == "__main__":
        
    # make directory to store datas
    os.makedirs(base_dir+'/grouped',exist_ok=True)
    
    # execute parallel process
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(parallel_function, arg) for arg in grouped_splits]
        
    # print results
    for future in as_completed(futures):
        print(future.result())