import random, string
import os, sys
import time
import argparse
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# CODE: set environment varaible
os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1000'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

import ray
from ray import train, tune
from ray.tune.progress_reporter import CLIReporter

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_bldg', type=int, default=12)
parser.add_argument('--selection', type=str, default='sequential')
parser.add_argument('--lookahead', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--filename_seed', type=int, default=int(time.time()))
parser.add_argument('--dtype', type=torch.dtype, default=torch.float64)
parser.add_argument('--epoch_size', type=int, default=25)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--savedir', type=str, default='/home/sbose/time-series-forecasting-federation/.logs')
args = parser.parse_args()

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

# CODE: imports
get_comstock = import_function('/home/sbose/time-series-forecasting-federation/files_for_appfl/comstock_dataloader.py','get_comstock')
mape = import_function('/home/sbose/time-series-forecasting-federation/files_for_appfl/metric.py','mape')
Transformer = import_function('/home/sbose/time-series-forecasting-federation/models/TRANSFORMER/TransformerN.py','Transformer')

# FUNCTION: set seed globally
def set_seed(seed=233):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# FUNCTION: generate a random string from an int
def generate_random_string(seed):
    random.seed(seed)
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=5))

# FUNCTION: m random numbers out of m, similar to np.random.choice
def generate_unique_random_numbers(seed, n, m):
    if m > n + 1:
        raise ValueError("m cannot be greater than n + 1 for unique numbers")
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(n + 1), size=m, replace=False).tolist()

# FUNCTION: get m random buildings
def get_pooled_datasets(
    num_bldg:int = 10,
    lookback:int = 12,
    lookahead:int = 4,
    selection:str = 'squential', # 'sequential' or 'random'
    dtype:torch.dtype = torch.float32,
    seed:int = 42,
    MAX_BLDG:int = 100000
):
    
    if selection == 'sequential':
        bldg_idx = np.array([i for i in range(num_bldg)])
    else:
        if selection == 'random':
            bldg_idx = generate_unique_random_numbers(seed, MAX_BLDG, num_bldg)
        else:
            raise ValueError('selection keyword contains an invalid value.')
        
    train_set, test_set = [], []
    
    for b in bldg_idx:
        t1, t2 = get_comstock(
            b,
            lookback,
            lookahead,
            dtype=dtype
        )
        train_set.append(t1)
        test_set.append(t2)
        
    train_set, test_set = ConcatDataset(train_set), ConcatDataset(test_set)
    
    return train_set, test_set

# FUNCTION: define the training function
def train_comstock(config):
    
    # create our data loader, model, and optimizer.
    step = 1
    train_set, test_set = get_pooled_datasets(
        num_bldg = args.num_bldg,
        lookback = config.get("lookback",12),
        lookahead = args.lookahead,
        selection = args.selection,
        seed = args.seed,
        dtype = args.dtype
    )
    train_dataloader = DataLoader(train_set,batch_size=config.get("BS",64),shuffle=False)
    test_dataloader = DataLoader(test_set,batch_size=64,shuffle=False)
    model = Transformer(
        x_size = 6,
        y_size = 1,
        u_size = 2,
        s_size = 7,
        lookback = config.get("lookback",12),
        lookahead = args.lookahead,
        d_model = config.get("model_dim", 64),
        e_layers = config.get("layers", 3),
        d_layers = config.get("layers", 3),
        dtype = args.dtype
    ).to(args.device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = optim.AdamW(
        model.parameters(),
        lr = config.get("lr",1e-4)
    )
    
    # run and report
    train_func(model, optimizer, train_dataloader)
    metric = test_func(model, test_dataloader)
    metrics = {'mape': metric}
    train.report(metrics)
        
# FUNCTION: train function for comstock
def train_func(model, optimizer, train_loader, device=args.device):
    model.train()
    batch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > args.epoch_size:
            print(f"[loss,step] : [{loss.item()},{batch}]")
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        batch = batch_idx
        
# FUNCTION: test function for comstock
def test_func(model, test_loader, device=args.device):
    model.eval()
    mapes = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            mapes.append(mape(target.cpu().numpy(),outputs.detach().cpu().numpy()))
    return sum(mapes) / len(mapes)
    
# MAIN
if __name__ == "__main__":
    
    # Set seed
    set_seed(args.seed)
    
    # Init
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    
    # Reporter
    reporter = CLIReporter(
        parameter_columns=["lr", "BS", "lookback", "model_dim", "layers"],
        metric_columns=["mape"]
    )
    
    # make folder
    os.makedirs(os.path.join(args.savedir,f'ray_{generate_random_string(args.filename_seed)}_lookahead_{args.lookahead}'), exist_ok=True)
    
    # Tuner
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_comstock),
            resources={'cpu':10*torch.cuda.device_count(),'gpu':torch.cuda.device_count()}
        ),
        run_config=train.RunConfig(
            name="pbt_test",
            stop={"mape": 2.5},
            storage_path=os.path.join(args.savedir,f'ray_{generate_random_string(args.filename_seed)}_lookahead_{args.lookahead}'),
            progress_reporter=reporter
        ),
        tune_config=tune.TuneConfig(
            num_samples=1
        ),
        param_space={
            "lookback": tune.grid_search([12, 48, 96]),
            "BS": tune.grid_search([16, 32, 64]),
            "lr": tune.grid_search([1e-5, 5e-5, 1e-4, 1e-3]),
            "model_dim": tune.grid_search([32, 64, 128]),
            "layers": tune.grid_search([3, 4])
        },
    )
    
    # Run
    results_grid = tuner.fit()
    
    # Extract best result
    best_result = results_grid.get_best_result(metric="mape", mode="min")
    print('Best result path:', best_result.path)
    print("Best final iteration hyperparameter config:\n", best_result.config)
    df = best_result.metrics_dataframe
    # Deduplicate, since PBT might introduce duplicate data
    df = df.drop_duplicates(subset="training_iteration", keep="last")
    df.plot("training_iteration", "mape")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.savefig(f"result_lookahead_{args.lookahead}.pdf",format="pdf",bbox_inches="tight")
    plt.close()
        
    