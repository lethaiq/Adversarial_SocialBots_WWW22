import gym
import numpy as np
import os
import random
import sys
import time
import torch

import ray
import glob
import tf_slim as slim

from gym.spaces import Box
from gym_bot.envs import AdvBotEnvSingleDetectLargeHiar
from ray import tune
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.visionnet import VisionNetwork as TorchConv

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.torch_ops import FLOAT_MAX
from ray.rllib.utils.torch_ops import FLOAT_MIN
from ray.tune import run_experiments
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
from tqdm import tqdm
from ge.graph_utils import *

SEED = 770
test_seed = 70
np.random.seed(SEED)
torch.manual_seed(SEED)
ENV_NAME = 'gym_bot:advbot-v6'
NAME = 'advbot-v6'
model_type = "CONV"
num_filters = 8
node_embed_dim = 6
WALK_P = 1
WALK_Q = 50
graph_algorithm = "node2vec"
train_probs = -1
test_probs = 0.25
mode = "gcn"
dataset = "train"
validation_graphs = {}

def load_graph():
    global validation_graphs
    print("loading ", dataset)
    if dataset == "original":
        files = glob.glob('./database/_hoaxy*.pkl')
        for file in files:
            try:
                validation_graph = pickle.load(open(file,'rb'))
                validation_graphs[file] = validation_graph
            except Exception as e:
                print(e)
                pass
    else:
        X_train, X_test = pickle.load(open('./database/split_hoaxy.pkl', 'rb'))
        print("X_train", len(X_train))
        # print("X_val", len(X_val))
        print("X_test", len(X_test))

        if dataset == "train":
            files = X_train
        # elif dataset == "val":
            # files = X_val
        else:
            files = X_test

        for file in files:
            try:
                if file not in validation_graphs:
                    validation_graph = pickle.load(open(file,'rb'))
                    validation_graphs[file] = validation_graph
                else:
                    print("ALREADY LOADED", file)
            except Exception as e:
                print("ERROR", e)
                pass
    
    print("Loaded ", len(validation_graphs), " real graphs")

def make_env(env_id, rank):
    def _init():
        env = gym.make(env_id, seed=SEED+rank)
        env.seed(SEED + rank)
        return env
    return _init


class TorchParametricActionsModel(DQNTorchModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 node_embed_dim=2,
                 true_obs_shape=(None, ),
                 activated_obs_shape=(None, ),
                 history_obs_shape=(None, ),
                 action_embed_size=None,
                 model_type="FCN",
                 num_filters=4,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)
        self.model_type = model_type
        self.vf_share_layers = True

        if model_type == "FCN":
            self.action_embed_model = TorchFC(
            Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size,
            model_config, name + "_action_embed")

        else:
            self.action_embed_model = TorchConv(
            Box(-1, 1, shape=true_obs_shape), action_space, int(action_embed_size/3),
            {"conv_filters": [[num_filters, [node_embed_dim,node_embed_dim], true_obs_shape[0]-1]]}, name + "_action_embed")

            self.activated_embed_model = TorchFC(
            Box(-1, 1, shape=activated_obs_shape), action_space, int(action_embed_size/3),
            model_config, name + "_activated_embed")

            self.history_embed_model = TorchFC(
            Box(-1, 1, shape=history_obs_shape), action_space, int(action_embed_size/3),
            model_config, name + "_history_embed")


    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embed, _ = self.action_embed_model({"obs": input_dict["obs"]["advbot"]})
        activated_embed, _ = self.activated_embed_model({"obs": input_dict["obs"]["activated"]})
        history_embed, _ = self.history_embed_model({"obs": input_dict["obs"]["history"]})
        features = torch.cat((action_embed, activated_embed, history_embed), 1)

        intent_vector = torch.unsqueeze(features, 1) #32, 1, 50
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function() + self.activated_embed_model.value_function() + self.history_embed_model.value_function()


def train(run_name=None):
    def env_creator(validation=False, validation_graphs=[], seed=77, reward_shaping=None):
        # assert validation == False, validation_graphs
        env = AdvBotEnvSingleDetectLargeHiar(seed=seed, 
                                            validation=validation,
                                            graph_algorithm=graph_algorithm.lower(),
                                            walk_p=WALK_P,
                                            walk_q=WALK_Q,
                                            model_type=model_type,
                                            node_embed_dim=node_embed_dim,
                                            probs=train_probs,
                                            mode=mode,
                                            validation_graphs=validation_graphs,
                                            reward_shaping=reward_shaping)
        return env

    register_env(NAME, lambda config: env_creator(**config))
    ModelCatalog.register_custom_model("pa_model", TorchParametricActionsModel)

    env = env_creator(validation=False)
    act_dim = env.action_dim
    obs_dim = env.level2_observation_space['advbot'].shape
    activated_dim = env.level2_observation_space['activated'].shape if model_type == "CONV" else None
    history_dim = env.level2_observation_space['history'].shape if model_type == "CONV" else None

    level2_model_config = {
        "model": {
            "custom_model": "pa_model",
            "custom_model_config": {"model_type": model_type,
                                    "true_obs_shape": obs_dim, 
                                    "action_embed_size": act_dim,
                                    "node_embed_dim": node_embed_dim,
                                    "num_filters": num_filters,
                                    "activated_obs_shape": activated_dim,
                                    "history_obs_shape": history_dim},
            "vf_share_layers": True,
        }}

    level1_model_config = {
        "model": {
            "use_lstm": True,
            "max_seq_len": 10,
            "lstm_cell_size": 16,
        },
    }
    level1_model_config = {}

    policy_graphs = {}
    policy_graphs['level1'] = (None, env.level1_observation_space, env.level1_action_space, level1_model_config)
    policy_graphs['level2'] = (None, env.level2_observation_space, env.level2_action_space, level2_model_config)

    def policy_mapping_fn(agent_id):
        return agent_id

    input_graphs = [validation_graphs[k] for k in validation_graphs]

    config={
        "log_level": "WARN",
        "num_workers": 5,
        "num_envs_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        "num_gpus": 1,
        
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },

        "lr": 0.0003,
        "entropy_coeff": 0.01,
        "seed": SEED,
        'framework': 'torch',
        "env": NAME
    }

    exp_dict = {
        'name': 'hierachical_synthetic6',
        "local_dir": os.environ.get('ADVBOT_LOG_FOLDER'),
        'run_or_experiment': 'PPO',
        "stop": {
            "training_iteration": 10000
        },
        'checkpoint_freq':5,
        "config": config,
        "callbacks": [WandbLoggerCallback(
            project="AdversarialBotsHierachicalSynthetic4-{}".format(graph_algorithm),
            group="User-{}".format(act_dim-1),
            api_key="5d4247fa5b879af8aeb0874889a94ca78d4be18d",
            log_config=True)]
    }

    ray.init()
    tune.run(**exp_dict)


def test(path, real_graph=0, custom_max_step=120, greedy=False, interval=None):
    print("INTERVAL", interval)

    global validation_graphs
    ray.init()
    load_graph()

    def env_creator(validation_graphs=[], seed=test_seed):
        env = AdvBotEnvSingleDetectLargeHiar(seed=test_seed, 
                                        validation=True,
                                        validation_graphs=validation_graphs,
                                        graph_algorithm=graph_algorithm.lower(),
                                        walk_p=WALK_P,
                                        walk_q=WALK_Q,
                                        model_type=model_type,
                                        node_embed_dim=node_embed_dim,
                                        probs=test_probs,
                                        mode=mode,
                                        custom_max_step=custom_max_step,
                                        interval=interval)
        print("VALIDATION:", env.validation)
        return env

    register_env(NAME, env_creator)
    ModelCatalog.register_custom_model("pa_model", TorchParametricActionsModel)
    env = env_creator()

    act_dim = env.action_dim
    obs_dim = env.level2_observation_space['advbot'].shape
    activated_dim = env.level2_observation_space['activated'].shape
    history_dim = env.level2_observation_space['history'].shape

    level2_model_config = {
        "model": {
            "custom_model": "pa_model",
            "custom_model_config": {"model_type": model_type,
                                    "true_obs_shape": obs_dim, 
                                    "action_embed_size": act_dim,
                                    "node_embed_dim": node_embed_dim,
                                    "num_filters": num_filters,
                                    "activated_obs_shape": activated_dim,
                                    "history_obs_shape": history_dim},
            "vf_share_layers": True
        }}
    policy_graphs = {}
    policy_graphs['level1'] = (None, env.level1_observation_space, env.level1_action_space, {})
    policy_graphs['level2'] = (None, env.level2_observation_space, env.level2_action_space, level2_model_config)

    def policy_mapping_fn(agent_id):
        return agent_id

    config={
        "log_level": "WARN",
        "num_workers": 5,
        "num_gpus": 1,
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "seed": test_seed + int(time.time()),
        'framework': 'torch',
        "env": NAME
    }

    agent = None
    agent = PPOTrainer(config=config, env=NAME)
    agent.restore(path)
    print("RESTORED CHECKPOINT")

    def get_action(obs, agent=None, env=None, greedy=None):
        action = {}
        if not greedy:
            explores = {
                'level1': False,
                'level2': False
            }
            action = {}
            for agent_id, agent_obs in obs.items():
                policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
                action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id, explore=explores[agent_id])

        else: #greedy
            assert env != None, "Need to provide the environment for greedy baseline"
            for agent_id, agent_obs in obs.items():
                if agent_id == "level1":
                    policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
                    action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id, explore=False)
                else:
                    action[agent_id] = env.next_best_greedy()

        return action


    total_rewards = []
    total_ts = []

    for name in validation_graphs:
        print("\nGRAPH: {}".format(name))
        graph = validation_graphs[name]
        env = env_creator(validation_graphs=[graph], seed=77777)
        count = {}
        done = False
        obs = env.reset()
        while not done:
            action = get_action(obs, agent, env=env, greedy=greedy)
            obs, reward, done, info = env.step(action)
            done = done['__all__']

        seeds = list(env.seed_nodes.keys())
        reward = env.cal_rewards(test=True, seeds=seeds, reward_shaping=None)
        reward = 1.0 * reward/env.best_reward()
        total_rewards.append(reward)
        total_ts.append(env.current_t)

        print("Action Sequence (First 10, Last 10):", env.state[:10], env.state[-10:])
        print("Number of Interaction:", len(env.state) - env.state.count("T"))
        print("Reward:", reward)
        # print("MAX_TIME", env.MAX_TIME_STEP)
        # print("HEURISTIC SEEDS", np.argsort(env.out_degree)[::-1][:env.MAX_TIME_STEP])
        # print("SELECTED SEEDS", seeds)

    print(total_rewards, np.mean(total_rewards), np.std(total_rewards))
    print(total_ts, np.mean(total_ts), np.std(total_ts))

if __name__ == '__main__':
    if sys.argv[1] == "train":
        train()

    elif sys.argv[1] == "test":
        test_probs = float(sys.argv[5])
        dataset = sys.argv[6]
        try:
            interval = int(sys.argv[7])
        except:
            interval = None
        test(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), greedy=False, interval=interval)
    else:
        print("GREEDY")
        test_probs = float(sys.argv[5])
        dataset = sys.argv[6]
        test(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), greedy=True)