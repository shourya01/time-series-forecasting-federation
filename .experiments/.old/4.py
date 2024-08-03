# Set up the basic prerequisites

import os
import numpy as np
import pandas as pd
import multiprocessing

base_dir = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA'
versionID = 32
cpu_cores = multiprocessing.cpu_count()
metadata = pd.read_parquet(base_dir+f'/upgrade{versionID}.parquet') 

# split into different meradatas depending upon which state it is from

metadata_grouped = metadata.groupby('in.state')
m_states = [group for _,group in metadata_grouped]

# split the metadata into list-of-lists which can be parallelized 

def chunk_list(lst, m):
    # function to chunk list almost equally into list-of-lists
    n = len(lst)
    avg = n / m
    out = []
    last = 0.0

    while last < n:
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out

m_states_parallel = chunk_list(m_states,cpu_cores)

# In this block, define the basic function which will allow parallelization

power_consumption_feature_name = 'out.electricity.total.energy_consumption'

def df_col_idx(idx,colname,df):
    
    # index a dataframe column as if it were a list
    # note: use SPARINGLY, can be expensive
    
    dfcol = df[colname].tolist()
    return dfcol[idx]

def parallel_function(m_states_superlist):
    
    for mdata in m_states_superlist:
        
        state_name = df_col_idx(0,'in.state',mdata)
        print(f'Executing state {state_name} on process PID {os.getpid()}.')
        
        # load all data
        bldg_ID = mdata.index.tolist()
        county_ID = mdata['in.nhgis_county_gisjoin'].tolist()
        
        corrs, retained_idx, avg_power = [], [], []
        ftr_name, ftr_name_recorded = [], False
        
        for bid,co_id in zip(bldg_ID,county_ID):
            
            if not (
                    os.path.exists(base_dir+f'/{state_name}/{bid}-{versionID}.parquet') 
                    and 
                    os.path.exists(base_dir+f'/weather/{co_id}_2018.csv')):
                print(f'Could not find building ID {bid} and/or weather data {co_id} for for state {state_name}. Continuing.')
                continue
            
            
            b_df = pd.read_parquet(base_dir+f'/{state_name}/{bid}-{versionID}.parquet')
            w_df = pd.read_csv(base_dir+f'/weather/{co_id}_2018.csv')
            retained_idx.append(bid)
            float_columns = b_df.select_dtypes(include=['float', 'float16', 'float32', 'float64']).columns.tolist()
            other_cols = [colname for colname in float_columns if colname != power_consumption_feature_name]
            weather_cols = w_df.select_dtypes(include=['float', 'float16', 'float32', 'float64']).columns.tolist()
            
            power_consumption = b_df[power_consumption_feature_name].to_numpy()
            avg_power.append(power_consumption.mean())
            
            other_features_corr = []
            
            # building specific features
            for ftr in other_cols:
                
                other_feature = b_df[ftr].to_numpy()
                with np.errstate(invalid='ignore'):
                    other_features_corr.append(np.corrcoef(power_consumption,other_feature)[0,1])
                if not ftr_name_recorded:
                    ftr_name.append(ftr)
                    
            # weather features
            for ftr in weather_cols:
                
                other_feature = np.repeat(w_df[ftr].to_numpy(),4)
                with np.errstate(invalid='ignore'):
                    other_features_corr.append(np.corrcoef(power_consumption,other_feature)[0,1])
                if not ftr_name_recorded:
                    ftr_name.append(ftr)
                    
            ftr_name_recorded = True
            
            corrs.append(other_features_corr)
            
        # take average of all correlations across buildings
        
        corrs_all_bldg = np.array(corrs).mean(axis=0)
        
        # descending order sort
        idx_desc = np.argsort(corrs_all_bldg)[::-1]
        feature_name_desc = [ftr_name[i] for i in idx_desc]
        corrs_desc = corrs_all_bldg[idx_desc]
        
        # before we print the results, we are also going to do a static features analysis
        # columns in the metadata which are numeric in value
        float_columns = mdata.select_dtypes(include=['float', 'float16', 'float32', 'float64']).columns.tolist()
        avg_power = np.array(avg_power)
        static_corr = []
        
        for ftr in float_columns:
            ftr_vec = mdata.loc[retained_idx, ftr].to_numpy()
            with np.errstate(invalid='ignore'):
                static_corr.append(np.corrcoef(avg_power,ftr_vec)[0,1])
        static_corr = np.array(static_corr)
        
        # organize static features in descending order
        # descending order sort
        idx_static_desc = np.argsort(static_corr)[::-1]
        feature_name_static_desc = [float_columns[i] for i in idx_static_desc]
        corrs_static_desc = static_corr[idx_static_desc]
        
        # open text file
        
        with open(base_dir+f'/corr_logs/{state_name}.txt','w') as file:
        
            # print correlations of dynamic features
            file.write(f'For state {state_name}, the dynamic feature correlation to the building power consumption, average across all buildings, are as follows.:\n\n')
            for fname,fval in zip(feature_name_desc,corrs_desc):
                if np.isnan(fval):
                    continue
                file.write(f"{fname}:\t {fval}\n")
                
            # print correlations of static features
            file.write(f'\nFor state {state_name}, the static feature correlation to the building power consumption, average across all buildings, are as follows.:\n\n')
            for fname,fval in zip(feature_name_static_desc,corrs_static_desc):
                if np.isnan(fval):
                    continue
                file.write(f"{fname}:\t {fval}\n")
                
        print(f'Finished saving correlation data for state {state_name}.')
            
    return 0

# main process

if __name__ == "__main__":
    
    # ensure that folder to save the logs are present
    
    os.makedirs(base_dir+'/corr_logs',exist_ok=True)
    
    # In this block, we achieve parallelization

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        results = list(executor.map(parallel_function, m_states_parallel))