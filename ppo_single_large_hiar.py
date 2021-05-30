from models import TorchParametricActionsModel
from test import test
from train import train
from utils import load_graph

import random
import sys
import torch
import numpy as np

SEED = 77
np.random.seed(SEED)
torch.manual_seed(SEED)

config = {
    "NAME":'advbot-v6',
    "run_name":None, 
    "seed":SEED, 
    "probs":0.8, #set -1 to random
    "graph_algorithm":"node2vec", 
    "WALK_P":1, 
    "WALK_Q":50, 
    "model_type":"CONV",
    "node_embed_dim":6,
    "num_filters":8,
    "validation_graphs":[],
    "reward_shaping":None,
    "num_workers":5,
    "num_gpus":1,
    "graph_feature":"gcn",
    "lr":0.0003,
    "entropy_coeff":0.01,
    "training_iteration":10000,
    "checkpoint_freq":5,
    "wandb_key":"5d4247fa5b879af8aeb0874889a94ca78d4be18d"
}

config_test = {
    "custom_max_step": 120,
    "detection_interval":20,
    "greedy": False,
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
        config["seed"] = 90

        test(model_path, **config, **config_test)