import argparse
from omegaconf import OmegaConf
from appfl.agent import ServerAgent
from appfl.communicator.grpc import GRPCServerCommunicator, serve
# import torch
# import numpy as np
# import random

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=42)
args = argparser.parse_args()

server_agent_config = OmegaConf.load("/home/sbose/time-series-forecasting-federation/.experiments/.configs/server.yaml")

# seed logic
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

server_agent = ServerAgent(server_agent_config=server_agent_config)

communicator = GRPCServerCommunicator(
    server_agent,
    max_message_size=server_agent_config.server_configs.comm_configs.grpc_configs.max_message_size,
    logger=server_agent.logger,
)

serve(
    communicator,
    **server_agent_config.server_configs.comm_configs.grpc_configs,
)