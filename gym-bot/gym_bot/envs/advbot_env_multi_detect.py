import gym
import numpy as np
from scipy import sparse
import os
import warnings
import math

from gym import error
from gym import spaces
from gym import utils
from gym.utils import seeding
from joblib import dump
from joblib import load
import torch
import time

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

# class AdvBotEnvMultiDetect(gym.Env): #advbotmulti-v0
class AdvBotEnvMultiDetect(MultiAgentEnv): #advbotmulti-v0
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "I"]
    ACTION_DICT = {"T":0, "I":1}
    FILTER_ACTION = ['F']
    MAX_TIME_STEP = 30
    HISTORY_LENGTH = 20
    MODEL_PATH = '{}/Documents/gym-advbots/traditional/RandomForestClassifier_stats_lengthNone.joblib'.format(os.path.expanduser("~"))

    COMPRESS_FOLLOWSHIP = True
    N_USERS = 100
    INTERVAL = 10
    RANDOM_FOLLOW = 3
    PROB_RETWEET = 0.25
    NORMAL_INTEREST = 1
    PENALTY_INTERACT = -0.1
    BONUS_DETECT = 0.1
    THREHOLD_INTERACT = 1
    VERBOSE = False

    def __init__(self, 
                num_bots=1, 
                discrete_history=False, 
                random_stimulation=True, 
                seed=77, 
                override={}, 
                validation=False,
                debug=False):
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

        self.action_encoder = OneHotEncoder(handle_unknown='ignore')
        self.action_encoder.fit(np.array(self.ACTION).reshape(-1,1))

        self.num_agents = num_bots

        self.n_fake_users = num_bots
        self.n_legit_users = self.N_USERS
        self.n_legit_tweets = 0
        self.n_fake_tweets = 1
        self.flg_fake_users = [0]*self.n_legit_users + [1]*self.n_fake_users

        self.total_tweets = self.n_legit_tweets + self.n_fake_tweets
        self.total_users = self.n_legit_users + self.n_fake_users

        self.timelog_key = "{}_{}"

        self.initialize()

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=self.pack_observation().shape)
        self.action_space = gym.spaces.Discrete(self.n_legit_users+1)

        self.detector = Detector(self.MODEL_PATH)
        
        if self.VERBOSE:
            print("loaded bot detector", self.detector.model)

    def initialize(self, reset_network=True):
        N = self.total_users
        M = self.total_tweets
        self.mat_timelines = np.zeros((M, N))
        self.mat_interaction = np.zeros((N, N))

        self.timelog = {}
        self.ownerlog = {}
        self.state = {}
        self.done = {}
        self.last_undetect = {}
        self.last_obs = {}

        for i in range(self.num_agents):
            self.state['bot_{}'.format(i)] = ""
            self.done['bot_{}'.format(i)] = 0
            self.last_undetect['bot_{}'.format(i)] = 0
            self.last_obs['bot_{}'.format(i)] = []

        self.global_current_t = 0
        self.global_previous_reward = 0
        
        if reset_network:
            self.mat_followship = np.zeros((N, N))
            self.stimulate_followship()
        else:
            self.mat_followship[self.n_legit_users,:] = 0
            self.mat_followship[:,self.n_legit_users] = 0


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
        if 'int' in str(type(user_indices)):
            user_indices = [user_indices]
        for j in user_indices:
            self.timelog[self.timelog_key.format(tweet_idx, j)] = self.global_current_t


    def pack_observation(self, bot_idx=None, share=True):
        followship = self.mat_followship.flatten().reshape(1, -1)
        timeline = self.mat_timelines.flatten().reshape(1, -1)

        def get_history(i):
            count_T = self.state['bot_{}'.format(i)].count("T")
            count_I = self.state['bot_{}'.format(i)].count("I")
            history = np.array([count_T, count_I]).reshape(1, -1)
            history = history / max(1, history.sum())
            return history

        if share:
            histories = []
            interactions = []
            for i in range(self.num_agents):
                history = get_history(i)
                histories.append(history)
            history = np.concatenate(histories, 1)
            interaction = self.mat_interaction[self.n_legit_users:].flatten().reshape(1, -1)
        else:
            assert bot_idx != None
            history = get_history(bot_idx)
            interaction = self.mat_interaction[self.n_legit_users+bot_idx:].flatten().reshape(1, -1)

        interaction = interaction / max(1, interaction.sum())
        # print(followship.min(), followship.max())
        # print(timeline.min(), timeline.max())
        # print(interaction.min(), interaction.max())
        # print(history.min(), history.max())
        obs = np.concatenate((followship, timeline, interaction, history), 1).flatten()
        # obs = (obs - obs.mean())/obs.std()
        return obs


    def reset(self, reset_network=True):
        self.initialize(reset_network=reset_network)
        global_obs = self.pack_observation()
        obs = {}
        for i in range(self.num_agents):
            obs['bot_{}'.format(i)] = global_obs
        return obs


    def update_retweet(self):
        tweet_idx = 0
        for i in range(self.n_legit_users):
            user_idx = i
            if self.mat_timelines[tweet_idx, user_idx]: #this user has a tweet to retweet
                origin_t = self.timelog[self.timelog_key.format(tweet_idx, user_idx)]
                delta_t = self.global_current_t - origin_t + 1
                prob = math.pow(self.PROB_RETWEET, delta_t)
                sample = np.random.binomial(1, prob)

                if sample == 1: #IF THIS USER RETWEETS
                    followers_idx = self.get_followers_idx(user_idx)
                    self.post_timeline(tweet_idx, followers_idx)


    def update_followship(self):
        for i in range(self.n_legit_users):
            user_idx = i
            has_interacts_idx = np.where(self.mat_interaction[:,user_idx])[0]
            for j in has_interacts_idx:
                if self.mat_followship[user_idx, j] == 0 and \
                self.mat_interaction[j, user_idx] >= self.THREHOLD_INTERACT:
                    self.mat_followship[user_idx, j] = 1
                    self.mat_interaction[user_idx, j] += self.NORMAL_INTEREST
        

    def get_followers_idx(self, user_idx):
        return np.where(self.mat_followship[:, user_idx])[0]


    def cal_rewards(self):
        for i in range(self.num_agents):
            bot_idx = self.n_legit_users + i
            tweet_idx = 0
            self.post_timeline(tweet_idx, bot_idx) 
            followers_idx = self.get_followers_idx(bot_idx)
            self.post_timeline(tweet_idx, followers_idx)
            if self.validation:
                iteration = 3
            else:
                iteration = 1
            for _ in range(iteration):
                self.update_followship()
                self.update_retweet()

        cur_reward = self.mat_timelines[0][:self.n_legit_users].sum()
        reward = cur_reward - self.global_previous_reward
        self.global_previous_reward = cur_reward
        return reward


    def render(self, mode=None, with_DNA=True):
        bot_idx = self.n_legit_users
        out = "\n\nTIMESTEP: {}".format(self.global_current_t)
        out += "\nTotal Interaction: {}".format(self.mat_interaction.sum())
        out += "\nTotal Followship: {}".format(self.mat_followship[:, bot_idx])
        print(out, end="", flush=True)


    def step_single(self, action, bot_idx):
        assert bot_idx != None
        action = action['bot_{}'.format(bot_idx)]
        add_reward = 0

        if action == 0:
            self.state['bot_{}'.format(bot_idx)] += "T"
            action = -1
        else:
            self.state['bot_{}'.format(bot_idx)] += "I"
            action = action - 1
            
            mat_idx = self.n_legit_users + bot_idx
            self.mat_interaction[mat_idx, action] += self.NORMAL_INTEREST
            if self.mat_interaction[mat_idx, action] > self.THREHOLD_INTERACT and \
            (self.mat_interaction[mat_idx,:] > 0).sum() < self.n_legit_users:
                add_reward += -self.PENALTY_INTERACT 

        return add_reward

    def step(self, action):
        obs, rew, info = {}, {}, {}

        self.global_current_t += 1
        add_reward = {}

        for i in range(self.num_agents):
            if 'bot_{}'.format(i) in action:
                add_reward['bot_{}'.format(i)] = self.step_single(action, i)

        self.update_followship()
        self.update_retweet()

        global_obs = self.pack_observation(share=True)
        influence_reward = self.cal_rewards()

        if self.global_current_t % self.INTERVAL == 0 and self.global_current_t > 0:
            for i in range(self.num_agents):
                if self.done['bot_{}'.format(i)] == 0:
                    pred = self.detector.predict(self.state['bot_{}'.format(i)])[0]
                    if pred >= 0.5:
                        self.done['bot_{}'.format(i)] = self.global_current_t
                        self.last_obs['bot_{}'.format(i)] = global_obs
                    else:
                        add_reward['bot_{}'.format(i)] += self.BONUS_DETECT * (self.global_current_t - self.last_undetect['bot_{}'.format(i)])
                        self.last_undetect['bot_{}'.format(i)] = self.global_current_t

        if self.global_current_t >= self.MAX_TIME_STEP:
            for i in range(self.num_agents):
                if self.done['bot_{}'.format(i)] == 0:
                    self.done['bot_{}'.format(i)] = self.global_current_t
                    self.last_obs['bot_{}'.format(i)] = global_obs
        
        for i in range(self.num_agents):
            if self.done['bot_{}'.format(i)] == 0 or (self.done['bot_{}'.format(i)] == self.global_current_t):
                temp_add_reward = 0
                if not self.validation: #only using additional rewards during training 
                    temp_add_reward = add_reward['bot_{}'.format(i)]
                rew['bot_{}'.format(i)] = influence_reward + temp_add_reward
                obs['bot_{}'.format(i)] = global_obs

        if sum([self.done[k] > 0 for k in self.done]) == self.num_agents:
            self.done["__all__"] = True
        else:
            self.done["__all__"] = False

        # print(self.global_current_t, obs.keys(), rew.keys(), self.done)
        return obs, rew, self.done, info


    def close(self):
        self.reset()
