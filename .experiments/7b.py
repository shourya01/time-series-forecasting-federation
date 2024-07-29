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

# Different models
from models.LSTM.LSTMFCDecoder import LSTMFCDecoder
from models.LSTM.LSTMAR import LSTMAR
from models.DARNN.DARNN import DARNN
from models.TRANSFORMER.TransformerAR import TransformerAR
from models.TRANSFORMER.Transformer import Transformer
from models.LOGTRANS.LogTransAR import LogTransAR
from models.INFORMER.Informer import Informer
from models.AUTOFORMER.Autoformer import Autoformer
from models.FEDFORMER.FedformerWavelet import FedformerWavelet
from models.FEDFORMER.FedformerFourier import FedformerFourier
from models.CROSSFORMER.Crossformer import Crossformer
from models.XLSTM.mLSTM import mLSTM

## read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="ComStock")

## model and metric
parser.add_argument("--model",
                    choices=[
                        'lstm_fc',
                        'lstm_ar',
                        'darnn',
                        'transformer_ar',
                        'transformer',
                        'logtrans_ar',
                        'informer',
                        'autoformer',
                        'fedformer_wavelet',
                        'fedformer_fourier',
                        'crossformer',
                        'mlstm'
                    ],
                    default='lstm_ar')
parser.add_argument("--dtype",
                    choices=[
                        'torch.float32',
                        'torch.float64'
                    ],
                    default='torch.float32')

## lookahead and lookback
parser.add_argument("--lookback", type=int, default=12)
parser.add_argument("--lookahead", type=int, default=8)

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
parser.add_argument("--server", type=str, default="ServerFedAdam")
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--server_lr", type=float, default=0.1)

## saving mode
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_frequency", type=int, default=5)

## parse
args = parser.parse_args()

# -----
# Function to determine if datasets are equal
# -----
def are_any_datasets_equal(datasets):
    n = len(datasets)
    for i in range(n):
        for j in range(i + 1, n):
            if len(datasets[i]) != len(datasets[j]):
                continue
            equal = all(compare_items(datasets[i][k], datasets[j][k]) for k in range(len(datasets[i])))
            if equal:
                return True
    return False

def compare_items(item1, item2):
    inp1, out1 = item1
    inp2, out2 = item2
    inp_res = all(torch.equal(i1,i2) for i1,i2 in zip(inp1,inp2))
    out_res = all(torch.equal(i1,i2) for i1,i2 in zip(out1,out2))
    return inp_res and out_res

# -----
# Dict for choosing model
# -----
model_name_dict = {
    'lstm_fc':LSTMFCDecoder,
    'lstm_ar':LSTMAR,
    'darnn':DARNN,
    'transformer_ar':TransformerAR,
    'transformer':Transformer,
    'logtrans_ar':LogTransAR,
    'informer':Informer,
    'autoformer':Autoformer,
    'fedformer_wavelet':FedformerWavelet,
    'fedformer_fourier':FedformerFourier,
    'crossformer':Crossformer,
    'mlstm':mLSTM
}
    
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
        print(f"\n\n-----\nCarrying out experiment ID {experimentID} with model {args.model} and {args.num_clients} clients.\nEvaluate test set status is: {args.do_validation}.\nRunning MPI {sync_kw}.\n-----\n")
    
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
    cfg.output_dirname = base_dir + "/.logs" + f"/outputs_{args.model}_{args.num_clients}clients_{args.server}_{args.num_epochs}epochs_validation_{args.do_validation}_mpi_{args.mpi_type}_expID_{experimentID}"
    
    ## User-defined model
    model = model_name_dict[args.model](
        x_size = 6,
        y_size = 1,
        u_size = 2,
        s_size = 7,
        lookback = args.lookback,
        lookahead = args.lookahead,
        dtype = eval(args.dtype)
    )   
    loss_fn = TSFLoss()
    metric = mape_metric
    
    ## User-defined data
    train_datasets, test_datasets = get_comstock(
        num_bldg = args.num_clients,
        lookback = args.lookback,
        lookahead = args.lookahead,
        dtype = eval(args.dtype)
    )
    
    ##
    
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
                
    ## If server process, ensure no test datasets are the same
    if comm_rank == 0:
        res = are_any_datasets_equal(test_datasets)
        if res:
            print(f"\n---\nDetected that test datasets are equal!\n---\n")
        else:
            print(f"\n---\nAll test datasets are unique!\n---\n")
    
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