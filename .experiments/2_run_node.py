import argparse
from omegaconf import OmegaConf
from appfl.agent import HFLNodeAgent
from appfl.communicator.grpc import GRPCHFLNodeServeCommunicator, GRPCHFLNodeConnectCommunicator, serve

argparser = argparse.ArgumentParser()
argparser.add_argument('--node_id', type=int, default=0)
args = argparser.parse_args()

hfl_node_agent_config = OmegaConf.load(f'/home/sbose/time-series-forecasting-federation/.experiments/.configs/node_{args.node_id+1}.yaml')

hfl_node_agent = HFLNodeAgent(hfl_node_agent_config=hfl_node_agent_config)
connect_communicator = GRPCHFLNodeConnectCommunicator(
    node_id=hfl_node_agent.get_id(),
    **hfl_node_agent_config.comm_configs.grpc_configs.connect,
)

serve_communicator = GRPCHFLNodeServeCommunicator(
    hfl_node_agent,
    connect_communicator=connect_communicator,
    max_message_size=hfl_node_agent_config.comm_configs.grpc_configs.serve.max_message_size,
    logger=hfl_node_agent.logger,
)

serve(
    serve_communicator,
    **hfl_node_agent_config.comm_configs.grpc_configs.serve,
)