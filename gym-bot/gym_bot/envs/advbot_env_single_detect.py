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

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.preprocessors import get_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence

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


class AdvBotEnvSingleDetect(gym.Env): #advbot-v4
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "I"]
    ACTION_DICT = {"T":0, "I":1}
    FILTER_ACTION = ['F']
    MAX_TIME_STEP = 30
    HISTORY_LENGTH = 20
    MODEL_PATH = '{}/Documents/gym-advbots/traditional/RandomForestClassifier_stats_lengthNone.joblib'.format(os.path.expanduser("~"))

    COMPRESS_FOLLOWSHIP = True
    N_USERS = 150
    INTERVAL = 10
    RANDOM_FOLLOW = 3
    PROB_RETWEET = 0.25
    NORMAL_INTEREST = 1
    THREHOLD_INTERACT = 1
    VERBOSE = False

    def __init__(self, 
                num_bots=1, 
                discrete_history=False, 
                random_stimulation=True, 
                seed=77, 
                override={}, 
                validation=False,
                debug=False,
                load_graph_path=None):
        self.seed(seed)

        for k in override:
            try:
                getattr(self, k)
                setattr(self, k, override[k])
                print("Update {} to {}".format(k, override[k]))
            except Exception as e:
                pass

        self.DEBUG = debug
        self.validation = validation
        self.discrete_history = discrete_history
        self.random_stimulation = random_stimulation

        self.n_fake_users = num_bots
        self.load_graph_path = load_graph_path

        if load_graph_path:
            self.graphs = []
            files = glob.glob('{}/*.gpickle'.format(load_graph_path))
            for file in files:
                g = nx.read_gpickle(file)
                self.graphs.append(nx.to_numpy_array(g))
            print("loaded {} graphs".format(len(self.graphs)))

        self.initialize()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.pack_observation().shape)
        self.action_space = gym.spaces.Discrete(self.n_legit_users+1)

        self.detector = Detector(self.MODEL_PATH)
        
        if self.VERBOSE:
            print("loaded bot detector", self.detector.model)

    def initialize(self, reset_network=True):
        if self.load_graph_path:
            i = np.random.choice(len(self.graphs))
            self.G = self.graphs[i]
            self.n_legit_users = len(self.G)
        else:
            self.n_legit_users = self.N_USERS

        self.flg_fake_users = [0]*self.n_legit_users + [1]*self.n_fake_users
        self.total_users = self.n_legit_users + self.n_fake_users
        self.total_tweets = 1
        N = self.total_users
        M = self.total_tweets
        self.mat_timelines = np.zeros((M, N))
        self.mat_interaction = np.zeros((N, N))
        self.timelog_key = "{}_{}"
        self.timelog = np.zeros(N)
        self.ownerlog = {}
        self.state = ""
        self.last_obs = {}
        self.last_rewards = {}
        self.done = 0
        self.current_t = 0
        self.curr_reward = 0
        self.previous_reward = 0
        self.last_undetect = 0
        
        if not self.load_graph_path:
            if reset_network:
                self.mat_followship = np.zeros((N, N))
                self.stimulate_followship()
            else:
                self.mat_followship[self.n_legit_users,:] = 0
                self.mat_followship[:,self.n_legit_users] = 0
        else:
            self.mat_followship = np.zeros((N, N))
            self.mat_followship[:self.n_legit_users,:self.n_legit_users] = self.G


    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def stimulate_followship(self):
        idx = np.random.choice(self.n_legit_users, (3+2+1)*self.RANDOM_FOLLOW, replace=False)
        idx1 = idx[0:3]
        idx2 = idx[3:5]
        idx3 = idx[5:6]
        for i in idx:
            self.mat_followship[i, idx2] = 1
            for j in idx2:
                self.mat_followship[j, idx3] = 1
    

    def post_timeline(self, tweet_idx, user_indices):
        self.mat_timelines[tweet_idx, user_indices] = 1
        assert 'int' not in str(type(user_indices))
            # user_indices = [user_indices]
        self.timelog[user_indices] = self.current_t


    def pack_observation(self):
        bot_idx = self.n_legit_users
        followship = self.mat_followship.flatten().reshape(1, -1)
        timeline = self.mat_timelines.flatten().reshape(1, -1)
        interaction = self.mat_interaction[bot_idx:].flatten().reshape(1, -1)
        count_T = self.state.count("T")
        count_I = self.state.count("I")
        history = np.array([count_T, count_I]).reshape(1, -1)
        history = history / max(1, history.sum())
        obs = np.concatenate((followship, timeline, interaction, history), 1).flatten()
        obs = (obs - obs.mean())/obs.std()
        return obs


    def reset(self, reset_network=True):
        self.initialize(reset_network=reset_network)
        obs = {}
        global_obs = self.pack_observation()
        return global_obs

    

    def update_retweet_single(self, user_idx, tweet_idx=0):
        def custombinomial(n,p):
            x = np.random.uniform()
            if x<p:
                return 1
            else:
                return 0  

        if self.mat_timelines[tweet_idx, user_idx]: #this user has a tweet to retweet
            origin_t = self.timelog[user_idx]
            delta_t = self.current_t - origin_t + 1
            prob = self.PROB_RETWEET**delta_t
            # sample = np.random.binomial(1, prob)
            sample = custombinomial(1, prob)

            if sample == 1: #IF THIS USER RETWEETS
                followers_idx = self.get_followers_idx(user_idx)
                self.post_timeline(tweet_idx, followers_idx)


    def update_retweet(self):
        for i in range(self.n_legit_users):
            self.update_retweet_single(i)


    def get_followers_idx(self, user_idx):
        return np.where(self.mat_followship[:, user_idx])[0]


    def cal_rewards(self):
        if self.validation:
            # N = self.total_users
            # M = self.total_tweets
            # self.mat_timelines = np.zeros((M, N))
            iteration = 3
        else:
            iteration = 1
            
        bot_idx = self.n_legit_users
        tweet_idx = 0
        self.post_timeline(tweet_idx, [bot_idx]) 
        followers_idx = self.get_followers_idx(bot_idx)
        self.post_timeline(tweet_idx, followers_idx)

        for _ in range(iteration):
            # self.update_followship()
            self.update_retweet()

        cur_reward = self.mat_timelines[0][:self.n_legit_users].sum()
        reward = cur_reward - self.previous_reward
        self.previous_reward = cur_reward
        return reward


    def render(self, mode=None, with_DNA=True):
        bot_idx = self.n_legit_users
        out = "\n\nTIMESTEP: {}".format(self.current_t)
        out += "\nTotal Interaction: {}".format(self.mat_interaction.sum())
        out += "\nTotal Followship: {}".format(self.mat_followship[:, bot_idx])
        print(out, end="", flush=True)


    def step(self, action):
        self.current_t += 1
        bot_idx = self.n_legit_users
        add_reward = 0

        if action == 0:
            self.state += "T"
            action = -1
        else:
            self.state += "I"
            action = action - 1
        
            self.mat_interaction[bot_idx, action] += self.NORMAL_INTEREST

            if self.mat_followship[action, bot_idx] == 0 and \
                self.mat_interaction[bot_idx, action] >= self.THREHOLD_INTERACT:
                    self.mat_followship[action, bot_idx] = 1
                    self.mat_interaction[action, bot_idx] += self.NORMAL_INTEREST

            if self.mat_interaction[bot_idx, action] > self.THREHOLD_INTERACT and \
            (self.mat_interaction[bot_idx,:] > 0).sum() < self.n_legit_users:
                add_reward += -0.1 


        # self.update_followship()
        self.update_retweet()

        if self.current_t % self.INTERVAL == 0 and self.current_t > 0:
            pred = self.detector.predict(self.state)[0]
            if pred >= 0.5:
                self.done = self.current_t
            else:
                add_reward += 0.1 * (self.current_t - self.last_undetect)
                self.last_undetect = self.current_t

        if self.current_t >= self.MAX_TIME_STEP:
            self.done = self.current_t
        global_obs = self.pack_observation()
        reward = self.cal_rewards()

        if not self.validation: #only using additional rewards during training 
            reward += add_reward

        obs, rew, done, info = global_obs, reward, self.done != 0, {}
        return obs, rew, done, info


    def close(self):
        self.reset()
