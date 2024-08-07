import argparse
from omegaconf import OmegaConf
from appfl.agent import ClientAgent
from appfl.communicator.grpc import GRPCClientCommunicator
# import numpy as np
# import torch 
# import random

argparser = argparse.ArgumentParser()
argparser.add_argument('--client_id', type=int, default=0)
argparser.add_argument('--seed', type=int, default=42)
args = argparser.parse_args()

client_agent_config = OmegaConf.load(f"/home/sbose/time-series-forecasting-federation/.experiments/.configs/client_{args.client_id+1}.yaml")

# set seed
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

client_agent = ClientAgent(client_agent_config=client_agent_config)
client_communicator = GRPCClientCommunicator(
    client_id = client_agent.get_id(),
    **client_agent_config.comm_configs.grpc_configs,
)

client_config = client_communicator.get_configuration()
client_agent.load_config(client_config)

init_global_model = client_communicator.get_global_model(init_model=True)
client_agent.load_parameters(init_global_model)

# Send the number of local data to the server
sample_size = client_agent.get_sample_size()
client_communicator.invoke_custom_action(action='set_sample_size', sample_size=sample_size)

while True:
    client_agent.train()
    local_model = client_agent.get_parameters()
    new_global_model, metadata = client_communicator.update_global_model(local_model)
    if metadata['status'] == 'DONE':
        break
    if 'local_steps' in metadata:
        client_agent.trainer.train_configs.num_local_steps = metadata['local_steps']
    client_agent.load_parameters(new_global_model)
client_communicator.invoke_custom_action(action='close_connection')
client_agent.clean_up()