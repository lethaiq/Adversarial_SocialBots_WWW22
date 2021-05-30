from utils import *
from train import train
from test import test
from models import TorchParametricActionsModel

import torch

SEED = 770
test_seed = 70
np.random.seed(SEED)
torch.manual_seed(SEED)
ENV_NAME = 'gym_bot:advbot-v6'
mode = "gcn"
dataset = "train"
validation_graphs = {}

if __name__ == '__main__':
    train(
    NAME='advbot-v6',
    run_name=None, 
    seed=77, 
    train_probs=-1,
    graph_algorithm="node2vec", 
    graph_feature="gcn",
    WALK_P=1, 
    WALK_Q=50, 
    model_type="CONV",
    node_embed_dim=6,
    num_filters=8,
    validation_graphs=[],
    reward_shaping=None,
    num_workers=5,
    num_gpus=1,
    lr=0.0003,
    entropy_coeff=0.01,
    training_iteration=10000,
    checkpoint_freq=5,
    wandb_key="5d4247fa5b879af8aeb0874889a94ca78d4be18d")