# This is the second download script after download1.ipynb
# In download1.ipynb, we had made subfolders corresponding to different states, and had also made a folder called weather.
# Each folder contains a file called links.txt.
# In this file, we attempt write script to actually download all the files.
# In the next script, we will actually process the data

import os
import s3fs
from mpi4py import MPI

state_names = ['KY', 'AZ', 'RI', 'SC', 'NH', 'WI', 'VT', 'MA', 'SD', 'KS', 'MS', 'IL', 'PA', 'NV', 'OR', 'MD', 'CA', 'ME', 'ND', 'CT', 'TN', 'GA', 'TX', 'NY', 'NC', 'IN', 'WV', 'CO', 'NE', 'ID', 'OK', 'DC', 'NJ', 'VA', 'WY', 'MO', 'MT', 'AK', 'IA', 'UT', 'MN', 'DE', 'MI', 'HI', 'AL', 'LA', 'WA', 'NM', 'AR', 'OH', 'FL']
sn_plus_weather = state_names + ['weather']
base_path = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA' # base path where the files are downloaded

def split_list(lst, n):
    # function to split elements of a list into n equal-ish parts
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

# main function
if __name__ == "__main__":
    
    # parallelization stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    print(f"On process {rank+1} of {world_size}.", flush=True)
    
    # split assignments
    task_splits = split_list(sn_plus_weather,world_size)
    assignment = list(task_splits)[rank]
    
    # execute assignments
    for task in assignment:
        
        # Complete tasks - either downloading state data files or weather data files.
            
        # set up s3fs downloader
        fs = s3fs.S3FileSystem(anon=True)
        bucket_name = 'oedi-data-lake'
        
        # current path
        cur_path = base_path + f'/{task}'
        
        # count number of lines in file
        ctr = 0
        with open(cur_path+'/links.txt', 'r') as file:
            for line in file:
                ctr += 1
        
        # set up paths and download 
        ctr_cur = 0
        with open(cur_path+'/links.txt', 'r') as file:
            for line in file:
                cur_line = line.strip()
                try:
                    if not os.path.exists(cur_path+'/'+os.path.basename(cur_line)): # check for existence
                        fstr = f's3://{bucket_name}/{cur_line}'
                        fs.get(fstr,cur_path)
                        exists_str = '(file already exists, not downloading)'
                    else:
                        exists_str = ''
                except Exception as e:
                    print(f"Download failed because of exception: {e} for file {fstr}.")
                ctr_cur += 1
                print(f'Process: {rank+1}/{world_size}, task: {task}, downloaded: {os.path.basename(cur_line)}, {ctr_cur}/{ctr} files. {exists_str}', flush=True)