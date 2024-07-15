# Installed module imports
import os, sys
import time, hashlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import argparse
from mpi4py import MPI
import cProfile
import pstats

# APPFL modules
from appfl.config import *
from appfl.misc.utils import *
import appfl.run_mpi as rm

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
    
    # Print:
    if comm_rank == 0:
        print(f"\n\n-----\nCarrying out experiment ID {experimentID} with {args.num_clients} clients. Evaluate test set status is: {args.do_validation}.\n-----\n")
    
    # process MPI
    args.num_clients = comm_size - 1 if args.num_clients <= 0 else args.num_clients
    
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
        input_size = 16,
        u_size = 2,
        hidden_size = 20,
        num_layers = 2,
        y_size = 1,
        fcnn_sizes = (20,10,10,1),
        activation = nn.ReLU,
        lookahead = 4,
        dtype = torch.float32
    )
    loss_fn = TSFLoss()
    metric = mape_metric
    
    ## User-defined data
    train_datasets, test_datasets = get_comstock(
        num_bldg = args.num_clients
    )
    
    ## Disable validation
    if not args.do_validation:
        cfg.validation = False
    
    ## Running
    if comm_rank == 0:
        rm.run_server(
            cfg,
            comm,
            model,
            loss_fn,
            args.num_clients,
            EmptyDataset(),
            args.dataset,
            metric,
        )
    else:
        rm.run_client(
            cfg,
            comm,
            model,
            loss_fn,
            args.num_clients,
            train_datasets,
            test_datasets if args.do_validation else EmptyDataset(),
            metric,
        )
        
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