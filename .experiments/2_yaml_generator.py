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
parser.add_argument('--overlap', type=int, default=0)
parser.add_argument('--model_dir', type=str, default='/home/exx/shourya/time-series-forecasting-federation/models')
parser.add_argument('--loss_dir', type=str, default='/home/exx/shourya/time-series-forecasting-federation/files_for_appfl')
parser.add_argument('--log_dir', type=str, default='/home/exx/shourya/time-series-forecasting-federation/.logs')
parser.add_argument('--experiment_dir', type=str, default='/home/exx/shourya/time-series-forecasting-federation/.experiments')
parser.add_argument('--server_lr', type=float, default=1e-2)
parser.add_argument('--client_lr', type=float, default=1e-5)
parser.add_argument('--normalize', type=int, default=0)
parser.add_argument('--node_replace', type=int, default=0)
parser.add_argument('--model_dim', type=int, default=64)
parser.add_argument("--model",
                    choices=[
                        'lstm_ar',
                        'darnn',
                        'transformer',
                        'transformern',
                        'logtrans',
                        'informer',
                        'autoformer',
                        'fedformer_fourier',
                        'crossformer',
                        'mlstm'
                    ],
                    default='lstm_ar')
parser.add_argument('--checkpoint', action='store_true')
parser.add_argument('--expID', type=str, default='MjQ3MDNmND')
args = parser.parse_args()

# add the modules and associated dicts/configurations
model_dir_dict = {
    'lstm_ar':'LSTM/LSTMAR.py',
    'darnn':'DARNN/DARNN.py',
    'transformer':'TRANSFORMER/Transformer.py',
    'transformern':'TRANSFORMER/TransformerN.py',
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
    'transformern': 'Transformer',
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
    'lookahead': 4,
    'd_model': args.model_dim
}

# functions to decide splits
def partition_with_overlap(n, m):
    '''
    n: num_clients
    m: num_overlaps    
    '''
    if m < 0 or m > n:
        raise ValueError("Invalid number of overlaps")

    # Split array into two parts
    arr = np.arange(n)
    left, right = np.array_split(arr, 2)
    
    # Initialize common elements array
    common = []

    # Add elements alternatively from left and right arrays to common
    for i in range(m):
        if i % 2 == 0 and len(left) > 0:
            common.append(left[-1])
            left = left[:-1]
        elif len(right) > 0:
            common.append(right[0])
            right = right[1:]

    # Convert to lists to avoid float conversion issue
    left = left.tolist()
    right = right.tolist()

    # Extend left and right with unique elements from common
    left.extend(np.unique(common))
    right = np.unique(common).tolist() + right
    
    unique_elements = sorted(set(left) | set(right))
    result = []

    for elem in unique_elements:
        if elem in left and elem in right:
            result.append([0, 1])
        elif elem in left:
            result.append(0)
        else:
            result.append(1)

    return result, (left,right)

# cuda function
def assign_cuda_devices(m):
    '''
    m: num_clients
    '''
    arr = np.arange(m)
    left, _ = np.array_split(arr, 2)
    
    result = ['cuda:0' if i in left else 'cuda:1' for i in arr]
    
    return result

# overlap string
def overlap_string(overlap):
    if overlap == 0:
        return 'NoOverlap'
    else:
        return f'Overlap{overlap}'

# dict for the fedavg server yaml file
fedavg_server_config = lambda num_clients: {
    'client_configs': {
        'train_configs': {
            # local trainer
            'trainer': 'NaiveTrainer',
            'mode': 'step',
            'num_local_steps': args.num_local_steps,
            'optim': 'SGD',
            'optim_args': {
                'lr': args.client_lr
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
            'train_batch_size': 32,
            'val_batch_size': 32,
            'train_data_shuffle': True,
            'val_data_shuffle': False
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
                'error_bound': 1,
                'param_cutoff': 1024
            }
        }
    },
    'server_configs': {
        'scheduler': 'SyncScheduler',
        'scheduler_kwargs': {
            'num_clients': num_clients,
            'same_init_model': True
        },
        'aggregator': 'FedAvgAggregator',
        'aggregator_kwargs': {
            'client_weights_mode': 'equal',
            'server_learning_rate': 1,
            'do_checkpoint': args.checkpoint,
            'checkpoint_dirname': os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/server'),
            'checkpoint_filename': 'model',
            'checkpoint_interval': 1,
            'replace': True
        },
        'device': 'cpu',
        'num_global_epochs': args.num_global_steps,
        'logging_output_dirname': os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/server'),
        'logging_output_filename': 'result',
        'comm_configs': {
            'grpc_configs': {
                'server_uri': 'localhost:50051',
                'max_message_size': 1048576,
                'use_ssl': False
            }
        }
    }
}

# lambda function for client yaml files
client_config = lambda cid, device, nodeID: {
    'client_id': f'Client{cid+1}',
    'train_configs': {
        'device': device,
        'logging_id': f'Client{cid+1}',
        'logging_output_dirname': os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/client_{cid+1}'),
        'logging_output_filename': 'result',
        'do_checkpoint': args.checkpoint,
        'checkpoint_dirname': os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/client_{cid+1}'),
        'checkpoint_filename': 'model',
        'checkpoint_interval': 1
    },
    'data_configs': {
        'dataset_path': os.path.join(args.loss_dir,'comstock_dataloader.py'),
        'dataset_name': 'get_comstock',
        'dataset_kwargs': {
            'bldg_idx': cid,
            'normalize': True if args.normalize==1 else False
        }
    },
    'comm_configs': {
        'grpc_configs': {
            'connect': [{
                'server_uri': f'localhost:{int(50051+nodeID+1)}',
                'max_message_size': 1048576,
                'use_ssl': False
            }] if not isinstance(nodeID,list) else [{
                'server_uri': f'localhost:{int(50051+this_nodeID+1)}',
                    'max_message_size': 1048576,
                    'use_ssl': False
            } for this_nodeID in nodeID]
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
    'aggregator_kwargs': {
        'do_checkpoint': args.checkpoint,
        'checkpoint_dirname': os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/node_{nodeID+1}'),
        'checkpoint_filename': 'model',
        'checkpoint_interval': 1,
        'server_learning_rate': args.server_lr,
        'replace': True if args.node_replace==1 else False
    },
    'device': 'cpu',
    'logging_output_dirname': os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/node_{nodeID+1}'),
    'logging_output_filename': 'result',
    'comm_configs': {
        'grpc_configs': {
            'connect': {
                'server_uri': 'localhost:50051',
                'max_message_size': 1048576,
                'use_ssl': False
            },
            'serve': {
                'server_uri': f'localhost:{int(50051+nodeID+1)}',
                'max_message_size': 1048576,
                'use_ssl': False
            }
        }
    }
}

# we only consider 2 nodes over here!
NUM_NODES = 2

# main
if __name__ == "__main__":
    
    # configuration path
    config_path = os.path.join(args.experiment_dir,'.configs')
    os.makedirs(config_path,exist_ok=True)
    
    # lists
    devices = assign_cuda_devices(args.num_clients)
    nodes, d_splits = partition_with_overlap(args.num_clients,args.overlap)
    print(f"Devices: {devices}")
    print(f"Nodelist: {nodes}")
    
    # Quit if 2 GPUs not found
    if torch.cuda.device_count() < NUM_NODES:
        raise ValueError(F"Need at least {NUM_NODES} cuda GPUs to carry out this experiment")
    
    # dump server yamls
    with open(os.path.join(config_path,'server.yaml'),'w') as file:
        yaml.dump(fedavg_server_config(NUM_NODES),file)
    os.makedirs(os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/server'),exist_ok=True)
    with open(os.path.join(os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/server'),'server.yaml'),'w') as file:
        yaml.dump(fedavg_server_config(NUM_NODES),file)
        
    # dump node yamls
    for nidx,d_split in enumerate(d_splits): # runs 2 times
        with open(os.path.join(config_path,f'node_{nidx+1}.yaml'),'w') as file:
            yaml.dump(node_config(nidx,len(d_split)),file)
        os.makedirs(os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/node_{nidx+1}'),exist_ok=True)
        with open(os.path.join(os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/node_{nidx+1}'),f'node_{nidx+1}.yaml'),'w') as file:
            yaml.dump(node_config(nidx,len(d_split)),file)
            
    # dump client yamls
    for cid, (device,node) in enumerate(zip(devices,nodes)):
        with open(os.path.join(config_path,f'client_{cid+1}.yaml'),'w') as file:
            yaml.dump(client_config(cid,device,node),file)
        os.makedirs(os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/client_{cid+1}'),exist_ok=True)
        with open(os.path.join(os.path.join(args.log_dir,f'{args.expID}_{args.model}_HFL{overlap_string(args.overlap)}_nodereplace_{args.node_replace}/client_{cid+1}'),f'client_{cid+1}.yaml'),'w') as file:
            yaml.dump(client_config(cid,device,node),file)