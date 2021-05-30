# ACORN

## Installation
Check ``req.txt`` file for details. Basically, we will need ``torch``, ``ray[rllib]``, ``tensorflow``, and other basic packages.  
Install the ``gym_bot`` environment:  
```
cd gym_bot
python -m pip install -e .
```

## Configurations
Check the ``ppo_single_large_hiar.py``.
```
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
```

## Train
RUN: ``python ppo_single_large_hiar.py train``
Example of Statistics on Synthetic Graphs. 
![Statistics on Synthetic Graphs](https://raw.githubusercontent.com/lethaiq/ACORN/main/resources/synthetic.png?token=ADJNWYT7SR4MDZULGAGCUHDAXUWJQ)
Example Statistics on Real Graphs. 
![Statistics on Real Graphs](https://raw.githubusercontent.com/lethaiq/ACORN/main/resources/real.png?token=ADJNWYQLGJQ7LSSBQKELLRTAXUWIW)


## Test
The checkpoint ``./checkpoint_best/checkpoint-150`` is the best checkpoint, result of which is resulted in the paper.  
RUN: ``python ppo_single_large_hiar.py test ./checkpoint_best/checkpoint-150``

Example outputs:
```
...
GRAPH: ./database/_hoaxy36.pkl
updating INTERVAL to  20
EVALUATING REAL GRPAH... 1500
[Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    0.0s finished
EVALUATING REAL GRPAH... 1500
[Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    0.0s finished
DONE HERE 120 120
out_degree [1 3 2 1 1] 120
Action Sequence (First 10, Last 10): MAMMRTMMRA MAMMRMAMMR
Number of Interaction: 124
Reward: 1.002000250031254
...
```
