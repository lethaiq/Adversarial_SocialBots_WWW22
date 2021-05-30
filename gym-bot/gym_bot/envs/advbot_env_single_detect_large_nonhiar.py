import gym
import numpy as np
from scipy import sparse
import os
import warnings
import math
import networkx as nx

from gym import error
from gym import spaces
from gym import utils
from gym.utils import seeding
from joblib import dump
from joblib import load
import torch
import time
import glob
from gym.spaces import Box, Discrete, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.preprocessors import get_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from keras.utils.np_utils import to_categorical   
from ge.gcn_test import *
from ge.graph_utils import *
import scipy.stats as ss

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Detector:
    def __init__(self, model_path):
        self.scaler, self.model = load(model_path) 

    def predict(self, action, follower=None, following=None):
        x = self.extract_features(action, follower, following)
        x = self.scaler.transform(x)
        return self.model.predict(x)

    def extract_features(self, action, follower=None, following=None):
        num_tweets = action.count('T')
        num_interactions = action.count('I')
        avg = num_interactions / max(1, num_tweets)
        return np.array([avg]).reshape(1, -1)


# class AdvBotEnvSingleDetectLargeNonHiar(MultiAgentEnv): #advbot-v6
class AdvBotEnvSingleDetectLargeNonHiar(gym.Env): #advbot-v6
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "I"]
    ACTION_DICT = {"T":0, "I":1}
    FILTER_ACTION = ['F']
    MAX_TIME_STEP = 100
    INTERVAL = 10
    OUT_DEGREE_MIN = 0
    MODEL_PATH = '{}/Documents/gym-advbots/traditional/RandomForestClassifier_stats_lengthNone.joblib'.format(os.path.expanduser("~"))

    ORIGIN_GRAPH = "copen"
    NUM_COMMUNITY = 100

    COMPRESS_FOLLOWSHIP = True
    VERBOSE = False

    def __init__(self, 
                num_bots=1, 
                discrete_history=False, 
                random_stimulation=True, 
                seed=77, 
                override={}, 
                validation=False,
                validation_graph=None,
                debug=False,
                graph_algorithm="node2vec",
                walk_p=1, 
                walk_q=1,
                flg_detection=False,
                model_type="FCN",
                node_embed_dim=2,
                probs=0.25):
        self.seed(seed)

        for k in override:
            try:
                getattr(self, k)
                setattr(self, k, override[k])
                print("Update {} to {}".format(k, override[k]))
            except Exception as e:
                pass

        self.graph_algorithm = graph_algorithm
        self.walk_p = walk_p
        self.walk_q = walk_q
        self.node_embed_dim = node_embed_dim
        self.flg_detection = flg_detection
        self.PROB_RETWEET = probs

        self.DEBUG = debug
        self.model_type = model_type
        self.validation = validation
        self.validation_graph = validation_graph
        self.discrete_history = discrete_history
        self.random_stimulation = random_stimulation

        self.n_fake_users = num_bots
        self.initialize()
        self.detector = Detector(self.MODEL_PATH)
        
        if self.VERBOSE:
            print("loaded bot detector", self.detector.model)


    def update_avail_actions(self):
        tweet_idx = 0
        activated_idx = np.array(list(self.seed_nodes.keys()))
        self.action_mask = np.array([1] * self.max_avail_actions) ## all actions are available
        if len(activated_idx) > 0:
            self.action_mask[activated_idx + 1] = 0

        single_node_idx = np.where(self.out_degree <= self.OUT_DEGREE_MIN)[0]
        self.action_mask[single_node_idx + 1] = 0

        if self.action_mask[1:].sum() == 0: # if there is no valid action => open all
            self.action_mask = np.array([1] * self.max_avail_actions)
            if len(activated_idx) > 0:
                self.action_mask[activated_idx + 1] = 0

        self.action_mask[0] = 1


    def vectorize_graph(self, g, mode="gcn"):
        if mode == "gcn":
            rt = np.stack(get_embeds(g, 
                    node_embed_dim=self.node_embed_dim,
                    alg=self.graph_algorithm,
                    p=self.walk_p, 
                    q=self.walk_q))

        elif mode == "out_degree":
            rt = self.out_degree/len(self.G)

        elif mode == "rank":
            rt = ss.rankdata(self.out_degree)/len(self.G)

        return rt


    def best_reward(self):
        idx = np.argsort(self.out_degree)[::-1][:self.MAX_TIME_STEP]
        cur_reward = self.compute_influence(self.G, list(idx), prob=self.PROB_RETWEET)
        return cur_reward


    def initialize(self, reset_network=True):
        if not (self.validation and self.validation_graph):
            self.G = randomize_graph(graph_name=self.ORIGIN_GRAPH, k=self.NUM_COMMUNITY, mode=3)
        else:
            self.G = self.validation_graph
            # print("LOADED validation graph of size:", len(self.G))

        self.out_degree = np.array([a[1] for a in list(self.G.out_degree(list(range(len(self.G)))))])
        # print("OUT_DEGREE", np.sort(self.out_degree)[::-1][:10])

        self.n_legit_users = len(self.G)
        self.max_avail_actions = self.n_legit_users + 1
        self.state = ""
        self.seed_nodes = {}
        # self.activated_idx = np.array([1]*self.n_legit_users)
        self.level1_reward = 0.0
        self.level2_reward = 0.0
        self.done = 0
        self.current_t = 0
        self.previous_reward = 0
        self.previous_rewards = []
        self.last_undetect = 0
        self.heuristic_optimal_reward = self.best_reward()

        self.G_obs = self.vectorize_graph(self.G, mode="out_degree")
        
        self.action_dim = self.node_embed_dim
        random_state = np.random.RandomState(seed=7777)
        self.action_assignments = random_state.normal(0, 1, (self.max_avail_actions, self.action_dim)).reshape(self.max_avail_actions, self.action_dim)
        
        # self.action_mapping = {}
        # random_order = list(range(self.max_avail_actions))
        # random.shuffle(random_order)
        # for i, order in enumerate(random_order):
        #     self.action_mapping[order] = i
        # self.action_assignments = self.action_assignments[random_order]

        self.update_avail_actions()

        self.action_space = gym.spaces.Discrete(self.max_avail_actions)
        temp_obs = self.pack_observation()
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.max_avail_actions, )),
            "avail_actions": Box(-10, 10, shape=(self.max_avail_actions, self.action_dim)),
            "advbot":  gym.spaces.Box(low=-10, high=10, shape=temp_obs['advbot'].shape),
        })


    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def pack_observation(self):
        count_T = self.state.count("T")
        count_I = self.state.count("I")
        history = np.array([count_T, count_I]).reshape(1, -1)
        history = history / max(1, self.MAX_TIME_STEP)
        network = self.G_obs

        activated_idx = np.array(list(self.seed_nodes.keys()))
        activated = np.array([0]*self.n_legit_users)
        if len(activated_idx):
            activated[activated_idx] = 1
        activated = activated.reshape(1, -1)

        if self.model_type == "FCN":
            network = network.flatten().reshape(1, -1)
            # if len(activated_idx):
            #     network[0,activated_idx] = 0
            advbot = np.concatenate((history, network, activated), 1)
            obs = {
                "action_mask": self.action_mask,
                "avail_actions": self.action_assignments,
                "advbot": advbot
            }

            # print("self.action_mask.shape", self.action_mask.shape)
            # print("self.avail_actions.shape", self.action_assignments.shape)
            # print("self.obs.shape", advbot.shape)

        else:
            obs = {
                "action_mask": self.action_mask,
                "avail_actions": self.action_assignments,
                "advbot": network.reshape(network.shape[0], network.shape[1], 1),
                "history": history
            }

        return obs


    def reset(self, reset_network=True):
        self.initialize(reset_network=reset_network)
        obs = self.pack_observation()
        return obs


    def compute_influence(self, graph, seed_nodes, prob, n_iters=5):
        total_spead = 0
        for i in range(n_iters):
            np.random.seed(i)
            active = seed_nodes[:]
            new_active = seed_nodes[:]
            while new_active:
                activated_nodes = []
                for node in new_active:
                    neighbors = list(graph.neighbors(node))
                    success = np.random.uniform(0, 1, len(neighbors)) < prob
                    activated_nodes += list(np.extract(success, neighbors))
                new_active = list(set(activated_nodes) - set(active))
                active += new_active
            total_spead += len(active)
        return total_spead / n_iters


    def cal_rewards(self, test=False, seeds=None, action=None, specific_action=False):
        if not seeds:
            seeds = list(self.seed_nodes.keys())

        if not test:
            if not specific_action:
                cur_reward = self.compute_influence(self.G, seeds, prob=self.PROB_RETWEET, n_iters=10)
                reward = cur_reward - self.previous_reward
                self.previous_reward = cur_reward
            else:
                assert action >= 0
                cur_reward = self.compute_influence(self.G, [action], prob=self.PROB_RETWEET, n_iters=10)
                reward = cur_reward
                
        else:
            print("SEEDS", seeds, len(seeds))
            print("out_degree", self.out_degree[seeds])
            cur_reward = self.compute_influence(self.G, seeds, prob=self.PROB_RETWEET, n_iters=10)
            reward = cur_reward

        # if reward > 4.0:
        #     print("ALERT!!!! {}-{}".format(test, reward), "action:", action, "probs", self.PROB_RETWEET, self.out_degree)

        return reward


    def render(self, mode=None):
        pass


    def shape_reward(self, reward, mode=1):
        heuristic = self.heuristic_optimal_reward

        if mode == 1:
            rt = 1.0 * reward/heuristic

        return rt


    def step(self, action):
        # action = self.action_mapping[action]
        self.current_t += 1
        detection_reward = 0.1
        if action == 0:
            self.state += "T"
            action = -1
            influence_reward = 0
        else:
            self.state += "I"
            action = action - 1
            self.seed_nodes[action] = 1
            influence_reward = self.cal_rewards(action=action, specific_action=True)
            influence_reward = self.shape_reward(influence_reward)

        if self.flg_detection:
            if self.current_t % self.INTERVAL == 0 and self.current_t > 0:
                pred = self.detector.predict(self.state)[0]
                if pred >= 0.5:
                    self.done = self.current_t
                else:
                    detection_reward += 0.1 * (self.current_t - self.last_undetect)
                    self.last_undetect = self.current_t

        if self.current_t >= self.MAX_TIME_STEP:
            self.done = self.current_t
            # seeds = list(self.seed_nodes.keys())
            # print("EPISODE SEEDS", seeds)
            # print("EPISODE SEEDS DEGREE", self.out_degree[seeds])
            # print("EPISODE REWARDS", self.previous_rewards)

        self.update_avail_actions()
        global_obs = self.pack_observation()

        # if not self.validation: #only using additional rewards during training 
        #     reward = influence_reward + detection_reward
        # else:
        reward = influence_reward
        self.previous_rewards.append(reward)

        obs, rew, done, info = global_obs, reward, self.done != 0, {}

        return obs, rew, done, info


    def close(self):
        self.reset()
