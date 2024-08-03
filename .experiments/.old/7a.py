# Installed module imports
import os, sys
import time, hashlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import argparse
from mpi4py import MPI
from torch.profiler import profile, record_function, ProfilerActivity
import cProfile
import pstats

# APPFL modules
from appfl.config import *
from appfl.misc.utils import *
import appfl.run_mpi as rm
import appfl.run_mpi_sync as rms

# Add custom dir
sys.path.insert(0,'/home/sbose/time-series-forecasting-federation')

# Custom module imports
from files_for_appfl.comstock_dataloader import get_comstock
from files_for_appfl.tsf_loss import TSFLoss
from files_for_appfl.tsf_metric import mape_metric
from models.LSTM.LSTMAR import LSTMAR

## read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="ComStock")

## mpi algorithm to run
parser.add_argument('--mpi_type', choices=['sync', 'nosync'], default='nosync')

## clients
parser.add_argument("--num_clients", type=int, default=-1)
parser.add_argument("--client_optimizer", type=str, default="Adam")
parser.add_argument("--client_lr", type=float, default=3e-3)
parser.add_argument(
    "--local_train_pattern",
    type=str,
    default="steps",
    choices=["steps", "epochs"],
    help="For local optimizer, what counter to use, number of steps or number of epochs",
)
parser.add_argument("--num_local_steps", type=int, default=10)
parser.add_argument("--num_local_epochs", type=int, default=2)
parser.add_argument("--do_validation", action="store_true")

## server
parser.add_argument("--server", type=str, default="ServerFedAvg")
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--server_lr", type=float, default=0.1)

## saving mode
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_frequency", type=int, default=5)

## parse
args = parser.parse_args()
    
# -----
# Hasing function
# -----

def hashing_fn():
    current_time = str(time.time())
    encoded_time = current_time.encode()
    hash_object = hashlib.sha256(encoded_time)
    return hash_object.hexdigest()[:10]

# -----
# Empty dataset for disabling test
# -----
class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("Empty dataset")
    
# -----
# Base directory
# -----
base_dir = '/home/sbose/time-series-forecasting-federation'
    
# -----
# run MAIN
# -----

def main(comm, comm_rank, comm_size, experimentID, base_dir):
    
    '''
    comm: MPI comm object
    comm_rank: comm process rank
    comm_size: comm process size
    experimentID: string denoting unique experiemnt ID
    base_dir: base directory corresponding to base git folder of experiments
    '''
    
    # process MPI
    args.num_clients = comm_size - 1 if args.num_clients <= 0 else args.num_clients
    
    # process MPI algorithm
    if args.mpi_type == 'sync':
        mpi_alg = rms
        assert args.num_clients == comm_size-1, f"Error: To run MPI sync with {args.num_clients} clients, need {args.num_clients+1} processes but got {comm_size}."
    else:
        mpi_alg = rm
        
    # Print:
    if comm_rank == 0:
        sync_kw = 'synchronously' if args.mpi_type == 'sync' else 'non-synchronously'
        print(f"\n\n-----\nCarrying out experiment ID {experimentID} with {args.num_clients} clients.\nEvaluate test set status is: {args.do_validation}.\nRunning MPI {sync_kw}.\n-----\n")
    
    # configuration
    cfg = OmegaConf.structured(Config)
    cfg.device = args.device
    cfg.reproduce = True
    set_seed(args.seed)
    
    # clients
    cfg.num_clients = args.num_clients
    cfg.fed.clientname = 'ClientStepOptim'
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_steps = args.num_local_steps
    
    # server
    cfg.fed.servername = args.server
    cfg.num_epochs = args.num_epochs
    
    ## outputs
    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = base_dir + "/.logs" + f"/outputs_{args.dataset}_{args.num_clients}clients_{args.server}_{args.num_epochs}epochs_validation_{args.do_validation}_expID_{experimentID}"
    
    ## User-defined model
    model = LSTMAR(
        x_size = 6,
        y_size = 1,
        u_size = 2,
        s_size = 7,
        lookback = 8,
        lookahead = 4   
    )   
    loss_fn = TSFLoss()
    metric = mape_metric
    
    ## User-defined data
    train_datasets, test_datasets = get_comstock(
        num_bldg = args.num_clients
    )
    
    ## Split datasets
    if comm_rank == 0:
        test_dset = EmptyDataset()
    else:
        if not args.do_validation:
            test_dset = EmptyDataset()
        else:
            if mpi_alg == rms:
                test_dset = test_datasets[comm_rank-1]
            else:
                test_dset = test_datasets
    
    ## Disable validation
    if not args.do_validation:
        cfg.validation = False
        
    ## Model save configuration
    if args.save_model:
        cfg.save_model = True
        cfg.save_model_state_dict = True
        cfg.save_model_dirname = cfg.output_dirname
        cfg.checkpoints_interval = args.save_frequency
    
    ## Running
    if comm_rank == 0:
        if mpi_alg == rm:
            alg_name = 'rm.run_server'
        else:
            alg_name = 'rms.run_server'
        eval(alg_name)(
            cfg,
            comm,
            model,
            loss_fn,
            args.num_clients,
            test_dset,
            args.dataset,
            metric,
        )
    else:
        kwargs_client = {
            'cfg': cfg,
            'comm': comm,
            'model': model,
            'loss_fn': loss_fn,
            'num_clients': args.num_clients,
            'train_data': train_datasets,
            'test_data': test_dset,
            'metric': metric
        }
        kwargs_client_sync = {k:v for k,v in kwargs_client.items() if k!='num_clients'}
        if mpi_alg == rm:
            alg_name = 'rm.run_client'
            eval(alg_name)(**kwargs_client)
        else:
            alg_name = 'rms.run_client'
            eval(alg_name)(**kwargs_client_sync)
        
        
    ## Sayonara
    print("------DONE------", comm_rank)
    
    
if __name__ == "__main__":
    
    # process MPI
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    assert (
        comm_size > 1
    ), "This script requires the toal number of processes to be greater than one!"
    
    # broadcast experiment ID
    if comm_rank == 0:
        expID = hashing_fn()
    else:
        expID = None
    expID = comm.bcast(expID, root=0)
    
    # make directory for storing profile and logs
    os.makedirs(base_dir+'/.logs',exist_ok=True)
    os.makedirs(base_dir+f'/.logs/cprofile_{expID}',exist_ok=True)
    cprofile_dir = base_dir+f'/.logs/cprofile_{expID}'
    
    # set up cProfile and enable
    profile = cProfile.Profile()
    profile.enable()
    
    # run main
    main(comm, comm_rank, comm_size, expID, base_dir)
    
    # disable cProfile and dump the stats
    profile.disable()
    stats = pstats.Stats(profile).sort_stats('time')
    with open(cprofile_dir+f'/process{comm_rank}.txt','w') as f:
        stats.stream = f
        stats.print_stats()