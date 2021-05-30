from models import TorchParametricActionsModel
from test import test
from train import train
from utils import load_graph

import random
import sys
import torch
import numpy as np

SEED = 77
test_seed = 90
np.random.seed(SEED)
torch.manual_seed(SEED)

config = {
    "NAME":'advbot-v6',
    "run_name":None, 
    "seed":SEED, 
    "probs":0.8, #set -1 to random
    "graph_algorithm":"node2vec", 
    "WALK_P":1, # parameter p of node2vec
    "WALK_Q":50, # parameter q of node2vec
    "model_type":"CONV", 
    "node_embed_dim":6, # node embedding dimension of node2vec
    "num_filters":8, # number of filters for CONV
    "validation_graphs":[],
    "reward_shaping":None, 
    "num_workers":5, # number of workers used during train/test
    "num_gpus":1, # number of GPUS
    "graph_feature":"gcn", # gcn means node2vec features
    "lr":0.0003, # learning rate
    "entropy_coeff":0.01, # ppo parameter
    "training_iteration":10000, # number of training iterations
    "checkpoint_freq":5, # frequency of saving checkpoints during training
    "wandb_key":"" #wandb API (replace with your own)
}

config_test = {
    "custom_max_step": 120, # we train on 60 timesteps be default but during test we test on longer 120
    "detection_interval":20, # interval K refered in the paper
    "greedy": False, # whether test the AgentI+H in the paper (heuristic method)
}

if __name__ == '__main__':
    if sys.argv[1] == "train":
        train(**config)

    else:
        model_path = sys.argv[2]
        test_graphs = load_graph("test")
        assert model_path != ""
        assert len(test_graphs)
        
        config["validation_graphs"] = test_graphs
        config["seed"] = test_seed

        test(model_path, **config, **config_test)