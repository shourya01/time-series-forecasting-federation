import os
import sys
import torch
import torch.nn as nn
import numpy as np
from mpi4py import MPI
import torch.nn.functional as F
from torch.utils.data import DataLoader
from time import time

sys.path.insert(0,'/home/sbose/time-series-forecasting-federation')
from models.LSTM.LSTMAR import LSTMAR
from models.LFDataset import LFDataset

# we first map data types here

dtype_map = {
    torch.float32: np.float32,
    torch.float: np.float32,
    torch.float64: np.float64,
    torch.double: np.float64,
    torch.float16: np.float16,
    torch.half: np.float16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.short: np.int16,
    torch.int32: np.int32,
    torch.int: np.int32,
    torch.int64: np.int64,
    torch.long: np.int64,
    torch.bool: np.bool_,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128
}
dtype_map_mpi = {
    'int8': MPI.INT8_T,
    'int16': MPI.INT16_T,
    'int32': MPI.INT32_T,
    'int64': MPI.INT64_T,
    'uint8': MPI.UINT8_T,
    'uint16': MPI.UINT16_T,
    'uint32': MPI.UINT32_T,
    'uint64': MPI.UINT64_T,
    'float32': MPI.FLOAT,
    'float64': MPI.DOUBLE,
    'complex64': MPI.COMPLEX,
    'complex128': MPI.DOUBLE_COMPLEX,
    # More mappings as needed
}

# some functions to flatten and unflatten state dicts to.from numpy vectors

def state_dict_to_vector(state_dict):
    param_vector = [param.detach().cpu().numpy().astype(dtype_map[param.dtype]).flatten() for param in state_dict.values()]
    return np.concatenate(param_vector)

def load_vector_to_state_dict(vector, reference_state_dict):
    pointer = 0
    with torch.no_grad():
        for param in reference_state_dict.values():
            num_param = param.numel()
            param.copy_(torch.from_numpy(vector[pointer:pointer + num_param]).view_as(param).to(param.dtype))
            pointer += num_param
    return reference_state_dict

def gradients_to_vector(state_dict):
    grad_vector = [(param.grad.detach().cpu().numpy().astype(dtype_map[param.dtype]).flatten() if param.grad is not None
                    else np.zeros(param.numel(), dtype=dtype_map[param.dtype]))
                   for param in state_dict.values()]
    return np.concatenate(grad_vector)

def load_vector_to_gradients(vector, reference_state_dict, dtype_map):
    pointer = 0
    with torch.no_grad():
        for param in reference_state_dict.values():
            num_elements = param.numel()
            part_vector = vector[pointer:pointer + num_elements]
            grad_tensor = torch.from_numpy(part_vector).view_as(param).to(dtype=dtype_map[param.dtype])
            if param.grad is not None:
                param.grad.copy_(grad_tensor)
            else:
                param.grad = grad_tensor.to(param.device)
            pointer += num_elements
    return reference_state_dict
            
# custom zero_grad implementation
def custom_zero_grad(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
            
# we now load the name of all the files

base_grouped_dir = '/lcrc/project/NEXTGENOPT/NREL_COMSTOCK_DATA/grouped'
suffix = '_data.npz'
filenames = []

for filename in os.listdir(base_grouped_dir):
    if filename.endswith(suffix):
        # Extract the leading characters and add them to the list
        filenames.append(filename[:-len(suffix)])

# model and device
device = 'cuda:0'
model = LSTMAR(
    input_size = 8,
    u_size = 2,
    hidden_size = 20,
    num_layers = 2,
    y_size = 1,
    fcnn_sizes = (20,10,10,1),
    activation = nn.ReLU,
    lookahead = 4,
    dtype = torch.float32
).to(device)
global_model_vec = state_dict_to_vector(model.state_dict())

# function to partition lists
def split_into_sublists(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

# learning rate
lr = 0.1

# main function

if __name__ == "__main__":
    
    init_time = time()
    
    # set up MPI
    comm = MPI.COMM_WORLD
    rk = comm.Get_rank()
    sz = comm.Get_size()
    
    # split files
    file_splits = list(split_into_sublists(filenames,sz))
    my_split = file_splits[rk]
    
    # for any MPI rank, load the global state vec into the model
    model.load_state_dict(load_vector_to_state_dict(global_model_vec,model.state_dict()))
    
    # if you're 0, then broadcast the global state vector, else receive it
    if rk == 0:
        pass
    else:
        global_model_vec = np.empty(global_model_vec.shape, dtype=global_model_vec.dtype)
    comm.Bcast(global_model_vec, root=0)
    
    # loop across filenames
    
    grads = []
    for f in my_split:
        
        # load the files
        data_y_s = np.load(base_grouped_dir+f'/{f}_data.npz')
        data_x_u = np.load(base_grouped_dir+f'/{f}_weather.npz')
        dset = LFDataset(
            data_y_s = data_y_s,
            data_x_u = data_x_u,
            lookback = 8,
            lookahead = 4,
            client_idx = 0,
            idx_x = [0,1],
            idx_u = [2,3],
            dtype = torch.float32
        )
        dload = DataLoader(dset, batch_size = 32, shuffle = True)
        
        # get next
        inp, out = next(iter(dload))
        
        # transfer to relevant devices
        inp = tuple([itm.to(device) for itm in inp])
        out = tuple([itm.to(device) for itm in out])
        
        # zero grad of model
        custom_zero_grad(model)
        
        # backprop
        out_model = model(inp)
        loss = F.mse_loss(out_model,out[1])
        loss.backward()
        
        # get grads
        grads.append(gradients_to_vector(model.state_dict()))
        
        # print
        print(f"On process {rk+1}/{sz}, finished processing {f}.")
        
    print(f"Finished backprops on {rk+1}/{sz}.")
        
    grads = np.array(grads, dtype=grads[0].dtype)
    grads_shapes_all = []
    
    if rk == 0:
        grads_shapes_all.append(int(grads.shape[0]))
        for i in range(1,sz):
            grads_shapes_all.append(comm.recv(source=i,tag=10))
        total_grads = sum(grads_shapes_all)
        inv = 1 / total_grads
    else:
        comm.send(int(grads.shape[0]),dest=0,tag=10)
        
    comm.Barrier()
    
    # now we receive the actual data
    
    if rk == 0:
        recvd_grads = [np.mean(grads,axis=0) * inv]
        for i in range(1,sz):
            empt = np.empty((global_model_vec.size,grads_shapes_all[i]), dtype=global_model_vec.dtype)
            comm.Recv([empt, dtype_map_mpi[global_model_vec.dtype.name]], source = i, tag = 59)
            recvd_grads.append(np.mean(grads,axis=0) * inv)
    else:
        comm.Send([grads, dtype_map_mpi[global_model_vec.dtype.name]], dest = 0, tag = 59)  
        
    comm.Barrier() 
        
    # finally, consolidate
    
    if rk == 0:
        grads0 = np.sum(np.array(recvd_grads),axis=0).astype(global_model_vec.dtype)
        global_model_vec -= lr * grads0
        final_time = time()
        print(f"We are at the server. One epoch took {final_time-init_time} seconds and covered {total_grads} clients.")
        