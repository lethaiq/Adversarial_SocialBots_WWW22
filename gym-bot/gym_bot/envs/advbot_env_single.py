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
        self.scaler, self.vectorizer, self.model = load(model_path) 

    def predict(self, action, follower, following):
        x_dna = self.vectorizer.transform([action])
        x_traditional = self.extract_features(action, follower, following)
        x = np.concatenate((x_traditional, x_dna.toarray()), 1)
        x = self.scaler.transform(x)
        return self.model.predict(x)

    def extract_features(self, action, follower, following):
        num_tweets = action.count('T')
        num_replies = action.count('A')
        num_retweets = action.count('R')
        num_mentions = action.count('M')

        avg_mentions_per_tweet = num_mentions / max(1, num_tweets)
        retweet_ratio = num_retweets / max(1, num_tweets)
        reply_ratio = num_replies / max(1, num_tweets)
        retweet_reply_ratio = num_retweets / max(1, num_retweets)

        follow_ratio = (1+follower)/(1+following)

        return np.array([num_tweets, num_replies, num_retweets, num_mentions, \
                avg_mentions_per_tweet, retweet_ratio, reply_ratio, \
                retweet_reply_ratio, follow_ratio]).reshape(1, -1)


class AdvBotEnvSingle(gym.Env):
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "R", "A", "M"]
    ACTION_DICT = {"T":0, "R":1, "A":2, "M":3}
    FILTER_ACTION = ['F']
    MAX_TIME_STEP = 20
    HISTORY_LENGTH = 32
    MODEL_PATH = '{}/Documents/gym-advbots/traditional/RandomForest_0.91CV1033.joblib'.format(os.path.expanduser("~"))

    COMPRESS_FOLLOWSHIP = True
    N_USERS = 100
    RANDOM_FOLLOW = 5
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
        self.discrete_history = discrete_history
        self.random_stimulation = random_stimulation

        self.action_encoder = OneHotEncoder(handle_unknown='ignore')
        self.action_encoder.fit(np.array(self.ACTION).reshape(-1,1))

        self.n_fake_users = num_bots
        self.n_legit_users = self.N_USERS
        self.n_legit_tweets = 0
        self.n_fake_tweets = 1
        self.flg_fake_users = [0]*self.n_legit_users + [1]*self.n_fake_users

        self.total_tweets = self.n_legit_tweets + self.n_fake_tweets
        self.total_users = self.n_legit_users + self.n_fake_users

        self.timelog_key = "{}_{}"

        self.initialize()

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.pack_observation().shape)
        self.action_space = gym.spaces.Discrete(self.n_legit_users)
        self.global_obs_length = len(self.pack_observation()) - len(self.pack_history()[0])
        self.history_obs_length = self.HISTORY_LENGTH

        self.detector = Detector(self.MODEL_PATH)
        
        if self.VERBOSE:
            print("loaded bot detector", self.detector.model)

    def initialize(self, reset_network=True):
        # self.seed(int(time.time()))
        N = self.total_users
        M = self.total_tweets
        self.mat_timelines = np.zeros((M, N))
        self.mat_interaction = np.zeros((N, N))
        self.timelog = {}
        self.ownerlog = {}
        self.DNA = "T"
        self.last_obs = {}
        self.last_rewards = {}
        self.done = 0
        self.current_t = 0
        self.curr_reward = 0
        self.previous_reward = 0
        
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
        # print(idx1, idx2, idx3)
        # idx = np.random.choice(self.n_legit_users, self.RANDOM_FOLLOW)
        for i in idx:
            # idx2 = np.random.choice(self.n_legit_users, self.RANDOM_FOLLOW)
            self.mat_followship[i, idx2] = 1
            for j in idx2:
                # idx3 = np.random.choice(self.n_legit_users, self.RANDOM_FOLLOW)
                self.mat_followship[j, idx3] = 1
        # print(idx, idx2, idx3)
            # for i in range(self.n_legit_users):
            #     if np.random.binomial(1, 0.5):
            #         to_follow = np.random.choice(self.n_legit_users, self.RANDOM_FOLLOW)
            #         self.mat_followship[i, to_follow] = 1
            #         for j in to_follow:
            #             if np.random.binomial(1, 0.5):
            #                 to_follow2 = np.random.choice(self.n_legit_users, self.RANDOM_FOLLOW)
            #                 self.mat_followship[j, to_follow2] = 1


    def post_timeline(self, tweet_idx, user_indices):
        self.mat_timelines[tweet_idx, user_indices] = 1
        if 'int' in str(type(user_indices)):
            user_indices = [user_indices]
        for j in user_indices:
            self.timelog[self.timelog_key.format(tweet_idx, j)] = self.current_t


    def pack_history(self):
        dna = list(self.DNA[-self.HISTORY_LENGTH:])
        if not self.discrete_history:
            dna = dna + ["PAD"]*(self.HISTORY_LENGTH - len(dna))
            pad_length = (self.HISTORY_LENGTH - len(dna))*len(self.ACTION)
            dna = self.action_encoder.transform(np.array(dna).reshape(-1,1)).toarray().reshape(1,-1)
        else:
            dna = list(self.DNA)
            dna = [[self.ACTION_DICT[a] for a in dna]]
        return dna


    def pack_observation(self):
        if not self.COMPRESS_FOLLOWSHIP:
            followship = self.mat_followship.flatten().reshape(1,-1) #10404
            timeline = self.mat_timelines.flatten().reshape(1, -1)
            action_history = self.pack_history()
            obs = np.concatenate((followship, timeline, action_history), 1).flatten()
        else:
            bot_idx = self.n_legit_users
            followship = self.mat_followship.flatten().reshape(1, -1)
            timeline = self.mat_timelines.flatten().reshape(1, -1)
            # action_history = self.pack_history()
            interaction = self.mat_interaction[bot_idx:].flatten().reshape(1, -1)
            # interaction = interaction/(1+interaction.sum())
            obs = np.concatenate((followship, timeline, interaction), 1).flatten()
            obs = (obs - obs.mean())/obs.std()
        return obs


    def reset(self, reset_network=True):
        self.initialize(reset_network=reset_network)
        obs = {}
        global_obs = self.pack_observation()
        return global_obs


    def update_retweet(self):
        tweet_idx = 0
        for i in range(self.n_legit_users):
            user_idx = i
            if self.mat_timelines[tweet_idx, user_idx]: #this user has a tweet to retweet
                origin_t = self.timelog[self.timelog_key.format(tweet_idx, user_idx)]
                delta_t = self.current_t - origin_t + 1
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


    def update_detection(self):
        if self.current_t == self.MAX_TIME_STEP: # KILL ALL BOTS AT MAX TIME STEP
            self.done = self.current_t
            # bot_idx = self.n_fake_users
            # if not self.done:
            #     dna = self.DNA
            #     for action in self.FILTER_ACTION:
            #         dna = dna.replace(action,'')
            #     follower = self.mat_followship[:,bot_idx].sum()
            #     following = self.mat_followship[bot_idx,:].sum()
            #     pred = self.detector.predict(dna, follower, following)[0]
            #     if pred >= 0.5:
            #         self.done = self.current_t
            #     else:
            #         self.done = -1

    def cal_rewards(self):
        bot_idx = self.n_legit_users
        tweet_idx = 0
        self.post_timeline(tweet_idx, bot_idx) 
        followers_idx = self.get_followers_idx(bot_idx)
        self.post_timeline(tweet_idx, followers_idx)
        self.update_followship()
        self.update_retweet()
        cur_reward = self.mat_timelines[0].sum()
        reward = cur_reward - self.previous_reward
        self.previous_reward = cur_reward
        return reward

    def cal_rewards_bk(self):
        if not self.done:
            reward = 0.0
        else:
            tweet_idx = 0
            bot_idx = self.n_legit_users
            self.post_timeline(tweet_idx, bot_idx) 
            followers_idx = self.get_followers_idx(bot_idx)
            self.post_timeline(tweet_idx, followers_idx)
            for _ in range(3):
                self.update_followship()
                self.update_retweet()
            reward = self.mat_timelines[0].sum()
        return reward


    def render(self, mode=None, with_DNA=True):
        bot_idx = self.n_legit_users
        out = "\n\nTIMESTEP: {}".format(self.current_t)
        out += "\nTotal Interaction: {}".format(self.mat_interaction.sum())
        out += "\nTotal Followship: {}".format(self.mat_followship[:, bot_idx])
        print(out, end="", flush=True)


    def step(self, action):
        bot_idx = self.n_legit_users
        
        self.mat_interaction[bot_idx, action] += self.NORMAL_INTEREST

        add_reward = 0.0
        if self.mat_interaction[bot_idx, action] > self.THREHOLD_INTERACT:
            add_reward = -0.1

        self.update_followship()
        self.update_retweet()
        self.update_detection()
        reward = self.cal_rewards()
        reward += add_reward
        global_obs = self.pack_observation()

        obs, rew, done, info = global_obs, reward, self.done != 0, {}

        self.current_t += 1

        return obs, rew, done, info


    def close(self):
        self.reset()