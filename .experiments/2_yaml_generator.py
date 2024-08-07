import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--num_clients', type=int, default=2)
parser.add_argument('--num_local_steps', type=int, default=10)
parser.add_argument('--num_global_steps', type=int, default=10)
parser.add_argument('--model_dir', type=str, default='/home/sbose/time-series-forecasting-federation/models')
parser.add_argument('--loss_dir', type=str, default='/home/sbose/time-series-forecasting-federation/files_for_appfl')
parser.add_argument('--log_dir', type=str, default='/home/sbose/time-series-forecasting-federation/.logs')
parser.add_argument('--experiment_dir', type=str, default='/home/sbose/time-series-forecasting-federation/.experiments')
parser.add_argument("--model",
                    choices=[
                        'lstm_ar',
                        'darnn',
                        'transformer',
                        'logtrans',
                        'informer',
                        'autoformer',
                        'fedformer_fourier',
                        'crossformer',
                        'mlstm'
                    ],
                    default='lstm_ar')
parser.add_argument('--expID', type=str, default='MjQ3MDNmND')
args = parser.parse_args()

# add the modules and associated dicts/configurations
model_dir_dict = {
    'lstm_ar':'LSTM/LSTMAR.py',
    'darnn':'DARNN/DARNN.py',
    'transformer':'TRANSFORMER/Transformer.py',
    'logtrans':'LOGTRANS/LogTrans.py',
    'informer':'INFORMER/Informer.py',
    'autoformer':'AUTOFORMER/Autoformer.py',
    'fedformer_fourier':'FEDFORMER/FedformerFourier.py',
    'crossformer':'CROSSFORMER/Crossformer.py',
    'mlstm':'XLSTM/mLSTM.py'
}
model_name_dict = {
    'lstm_ar':'LSTMAR',
    'darnn':'DARNN',
    'transformer':'Transformer',
    'logtrans':'LogTrans',
    'informer':'Informer',
    'autoformer':'Autoformer',
    'fedformer_fourier':'FedformerFourier',
    'crossformer':'Crossformer',
    'mlstm':'mLSTM'
}
model_kwargs = {
    'x_size': 6,
    'y_size': 1,
    'u_size': 2,
    's_size': 7,
    'lookback': 12,
    'lookahead': 4
}

# dict for the fedavg server yaml file
fedavg_server_config = {
    'client_configs': {
        'train_configs': {
            # local trainer
            'trainer': 'NaiveTrainer',
            'mode': 'step',
            'num_local_steps': args.num_local_steps,
            'optim': 'SGD',
            'optim_args': {
                'lr': 1e-4
            },
            # loss function
            'loss_fn_path': os.path.join(args.loss_dir,'loss.py'),
            'loss_fn_name': 'MSELoss',
            # validation
            'do_validation': True,
            'do_pre_validation': False,
            'metric_path': os.path.join(args.loss_dir,'metric.py'),
            'metric_name': 'mape',
            # data loader
            'train_batch_size': 64,
            'val_batch_size': 64,
            'train_data_shuffle': True,
            'val_data_shuffle': True
        },
        'model_configs': {
            'model_path': os.path.join(args.model_dir,model_dir_dict[args.model]),
            'model_name': model_name_dict[args.model],
            'model_kwargs': model_kwargs
        },
        'comm_configs': {
            'compressor_configs': {
                'enable_compression': False,
                'lossy_compressor': 'SZ2Compressor',
                'lossless_compressor': 'blosc',
                'error_bounding_model': 'REL',
                'error_bound': 1e-3,
                'param_cutoff': 1024
            }
        }
    },
    'server_configs': {
        'scheduler': 'SyncScheduler',
        'scheduler_kwargs': {
            'num_clients': args.num_clients,
            'same_init_model': True
        },
        'aggregator': 'FedAvgAggregator',
        'aggregator_kwargs': {
            'client_weights_mode': 'equal',
            'server_learning_rate': 5e-3
        },
        'device': 'cpu',
        'num_global_epochs': args.num_global_steps,
        'logging_output_dirname': os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFLNoOverlap/server'),
        'logging_output_filename': 'result',
        'comm_configs': {
            'grpc_configs': {
                'server_uri': 'localhost:51051',
                'max_message_size': 1048576,
                'use_ssl': False
            }
        }
    }
}

# lambda function for client yaml files
client_config = lambda cid, device, numNodes: {
    'train_configs': {
        'device': device,
        'logging_id': f'Client{cid+1}',
        'logging_output_dirname': os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFLNoOverlap/client_{cid+1}'),
        'logging_output_filename': 'result'
    },
    'data_configs': {
        'dataset_path': os.path.join(args.loss_dir,'comstock_dataloader.py'),
        'dataset_name': 'get_comstock',
        'dataset_kwargs': {
            'bldg_idx': cid
        }
    },
    'comm_configs': {
        'grpc_configs': {
            'server_uri': f'localhost:{int(51051+cid+numNodes+1)}',
            'max_message_size': 1048576,
            'use_ssl': False
        }
    }
}

# lambda function to configure nodes
node_config = lambda nodeID, num_clients: {
    'node_id': f'Node{nodeID}',
    'scheduler': 'SyncScheduler',
    'scheduler_kwargs': {
        'num_clients': num_clients
    },
    'aggregator': 'HFLFedAvgAggregator',
    'device': 'cpu',
    'logging_output_dirname': os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFLNoOverlap/node_{nodeID+1}'),
    'logging_output_filename': 'result',
    'comm_configs': {
        'grpc_configs': {
            'connect': {
                'server_uri': 'localhost:51051',
                'max_message_size': 1048576,
                'use_ssl': False
            },
            'serve': {
                'server_uri': f'localhost:{int(51051+nodeID+1)}',
                'max_message_size': 1048576,
                'use_ssl': False
            }
        }
    }
}

# main
if __name__ == "__main__":
    
    num_clients = args.num_clients
    
    # configuration path
    config_path = os.path.join(args.experiment_dir,'.configs')
    os.makedirs(config_path,exist_ok=True)
    
    # split gpus among clients
    devices, nodes = [], []
    if torch.cuda.device_count() > 0:
        device_splits = np.array_split(np.arange(num_clients),torch.cuda.device_count())
        # couple hfl node logic with gpu assignment logic
        for sidx, split in enumerate(device_splits):
            with open(os.path.join(config_path,f'node_{sidx+1}.yaml'),'w') as file:
                yaml.dump(node_config(sidx,num_clients),file)
        for cid in range(num_clients):
            for sidx, split in enumerate(device_splits):
                if cid in split:
                    devices.append(f'cuda:{sidx}')
                    nodes.append(sidx)
    else:
        for cid in range(num_clients):
            nodes.append(0)
            devices.append('cpu')
            
    # ensure that relevant directories are present
    os.makedirs(os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFLNoOverlap/server'),exist_ok=True)
    for cid in range(num_clients):
        os.makedirs(os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFLNoOverlap/client_{cid+1}'),exist_ok=True)
    
    # generate and save the yaml files
    # server
    with open(os.path.join(config_path,'server.yaml'),'w') as file:
        yaml.dump(fedavg_server_config,file)
    # clients 
    for cid in range(num_clients):
        with open(os.path.join(config_path,f'client_{cid+1}.yaml'),'w') as file:
            yaml.dump(client_config(cid,devices[cid],torch.cuda.device_count()),file)