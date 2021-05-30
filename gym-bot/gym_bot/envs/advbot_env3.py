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


class AdvBotEnvFull(gym.Env):
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "R", "A", "M", 'F', 'PAD']
    ACTION_DICT = {"T":1, "R":2, "A":3, "M":4, "F":5, "PAD":0}
    FILTER_ACTION = ['F']
    MAX_TIME_STEP = 50
    HISTORY_LENGTH = 25
    INTERVAL = 64
    MODEL_PATH = '{}/Documents/gym-advbots/traditional/RandomForest_0.91CV1033.joblib'.format(os.path.expanduser("~"))

    RANDOM_FOLLOW = 3
    PROB_RETWEET = 0.25
    FOLLOW_INTEREST = 10
    NORMAL_INTEREST = 1
    THREHOLD_INTERACT = 11
    THREHOLD_STOP = 1
    LIVE_INCENTIVE = 0.0001
    VERBOSE = False

    def __init__(self, num_bots=1):
        self.n_actions = 6
        self.action_encoder = OneHotEncoder()
        self.action_encoder.fit(np.array(self.ACTION).reshape(-1,1))

        self.n_agents = num_bots
        self.n_fake_users = num_bots
        self.n_legit_users = 100
        self.n_legit_tweets = 0
        self.n_fake_tweets = 1
        self.flg_fake_users = [0]*self.n_legit_users + [1]*self.n_fake_users
        # self.flg_fake_tweets = [0]*self.n_legit_tweets + [1]+self.n_fake_tweets

        self.total_tweets = self.n_legit_tweets + self.n_fake_tweets
        self.total_users = self.n_legit_users + self.n_fake_users

        self.key = "bot_{}"
        self.timelog_key = "{}_{}"

        self.initialize()

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.pack_observation().shape)
        self.action_space = gym.spaces.MultiDiscrete([self.n_actions, self.total_users])
        self.agents = [self.key.format(i) for i in range(self.n_fake_users)]
        self.global_obs_length = len(self.mat_followship.flatten()) + len(self.mat_timelines.flatten())
        self.history_obs_length = self.HISTORY_LENGTH

        # print("self.global_obs_length", self.global_obs_length)
        # print("self.history_obs_length", self.history_obs_length)

        self.detector = Detector(self.MODEL_PATH)
        
        if self.VERBOSE:
            print("loaded bot detector", self.detector.model)


    def initialize(self):
        N = self.total_users
        M = self.total_tweets
        self.mat_timelines = np.zeros((M, N))
        self.mat_followship = np.zeros((N, N))
        self.mat_interaction = np.zeros((N, N))
        self.timelog = {}
        self.ownerlog = {}
        self.DNA = {}
        self.last_obs = {}
        self.last_rewards = {}
        self.done = {}
        self.current_t = 0
        self.curr_reward = 0

        for i in range(self.n_fake_users):
            self.DNA[self.key.format(i)] = "T"
            self.done[self.key.format(i)] = 0

        self.stimulate_followship()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def stimulate_followship(self):
        level1 = np.array(range(0, self.n_legit_users, 5))
        level2 = np.array(range(0, self.n_legit_users, 8))
        level3 = np.array(range(0, self.n_legit_users, 17))

        for i in level1:
            self.mat_followship[i, level2] = 1
        for j in level2:
            self.mat_followship[j, level3] = 1

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
        rt = []
        for i in range(self.n_fake_users):
            dna = list(self.DNA[self.key.format(i)][-self.HISTORY_LENGTH:])
            dna = dna + ["PAD"]*(self.HISTORY_LENGTH - len(dna))
            dna = [self.ACTION_DICT[a] for a in dna]
            # pad_length = (self.HISTORY_LENGTH - len(dna))*len(self.ACTION)
            # dna = self.action_encoder.transform(np.array(dna).reshape(-1,1)).toarray().reshape(1,-1)
            # if len(dna) < len(self.ACTION)*self.HISTORY_LENGTH:
            #     padding = np.array([[0]*pad_length])
            #     dna = np.concatenate((dna, padding),1)
            rt.append(dna)

        if len(rt) > 1:
            return np.concatenate(rt, 1)
        else:
            return rt


    def pack_observation(self):
        followship = self.mat_followship.flatten().reshape(1,-1) #10404
        timeline = self.mat_timelines.flatten().reshape(1, -1)
        action_history = self.pack_history()
        # print(followship.shape, action_history.shape)
        obs = np.concatenate((followship, timeline, action_history), 1).flatten()
        return obs

        

    def reset(self):
        self.initialize()
        obs = {}
        global_obs = self.pack_observation()
        for i in range(self.n_fake_users):
            obs[self.key.format(i)] = global_obs
        return obs


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
                    before_reward = self.mat_timelines[0].sum()
                    self.post_timeline(tweet_idx, followers_idx)
                    after_reward = self.mat_timelines[0].sum()

                    if self.VERBOSE:
                        print("delta_t, PROB, SAMPLE", delta_t, prob, sample)
                        print("cascading from user {} to followers".format(i), 
                            followers_idx, " +", after_reward - before_reward)


    def update_followship(self):
        for i in range(self.n_legit_users):
            user_idx = i
            has_interacts_idx = np.where(self.mat_interaction[:,user_idx])[0]
            for j in has_interacts_idx:
                if self.mat_followship[user_idx, j] == 0 and \
                self.mat_interaction[j, user_idx] > self.THREHOLD_INTERACT:
                    self.mat_followship[user_idx, j] = 1
                    self.mat_interaction[user_idx, j] += self.FOLLOW_INTEREST
        

    def get_followers_idx(self, user_idx):
        return np.where(self.mat_followship[:, user_idx])[0]


    def update_detection(self):
        if self.current_t >= self.MAX_TIME_STEP: # KILL ALL BOTS AT MAX TIME STEP
            for i in range(self.n_fake_users):
                self.done[self.key.format(i)] = self.current_t

        # elif self.current_t % self.INTERVAL == 0 and self.current_t > 1:
        #     for i in range(self.n_fake_users):
        #         bot_idx = i + self.n_fake_users
        #         if not self.done[self.key.format(i)]:
        #             dna = self.DNA[self.key.format(i)]
        #             for action in self.FILTER_ACTION:
        #                 dna = dna.replace(action,'')

        #             follower = self.mat_followship[:,bot_idx].sum()
        #             following = self.mat_followship[bot_idx,:].sum()
        #             pred = self.detector.predict(dna, follower, following)[0]
        #             if pred >= 0.5:
        #                 self.done[self.key.format(i)] = self.current_t


    def check_all_killed(self):    
        all_bot_killed = True if np.sum(self.done[i] != 0 for i in self.done) == self.n_fake_users else False
        return all_bot_killed


    def cal_rewards(self):
        reward = self.LIVE_INCENTIVE
        reward = 0
        if self.check_all_killed():
            delta = 999
            reward = self.mat_timelines[0].sum()
            while delta > self.THREHOLD_STOP:
                self.update_retweet()
                self.update_followship()
                curr_reward = self.mat_timelines[0].sum()
                delta = curr_reward - reward
                self.current_t += 1
                reward = curr_reward

                if self.VERBOSE:
                    print("Delta", delta)

            # reward = reward/self.n_fake_users + self.LIVE_INCENTIVE/self.n_fake_users #give incentives to live
            reward = reward + self.LIVE_INCENTIVE

        self.curr_reward = reward
        return reward


    def render(self, with_DNA=False):
        out = "\n\nTIMESTEP: {}".format(self.current_t)
        out += "\nCurrent Reward: {}/{}".format(\
            round(self.curr_reward, 2), self.n_legit_users)
        out += "\nTotal Followship: {}".format(self.mat_followship.sum())
        out += "\nTotal Interaction: {}".format(self.mat_interaction.sum())
        for i in range(self.n_fake_users):
            bot_idx = i + self.n_legit_users
            out += "\nBot : {} AGE {}".format(i, self.done[self.key.format(i)])
            out += "\n=> # Followees: {}".format(self.mat_followship[bot_idx,:].sum())
            out += "\n=> # Followers: {}".format(self.mat_followship[:, bot_idx].sum())
            if with_DNA:
                out += "\n=> DNA of Bot: {}".format(self.DNA[self.key.format(i)])
        print(out, end="", flush=True)


    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        for i in range(self.n_fake_users):
            if self.done[self.key.format(i)]:
                continue

            bot_idx = i + self.n_legit_users
            action = action_dict[self.key.format(i)]
            pp = get_preprocessor(self.action_space)(self.action_space)
            action = pp.transform(action)
            action_level1 = np.argmax(action[:self.n_actions])
            action_level2 = np.argmax(action[self.n_actions:])

            if action_level1 == 0: #TWEET
                tweet_idx = 0
                self.post_timeline(tweet_idx, bot_idx) # post the tweet on the current bot's timeline
                followers_idx = self.get_followers_idx(bot_idx)
                self.post_timeline(tweet_idx, followers_idx) # post the tweet on followers' timelines
                self.DNA[self.key.format(i)] += "T"

            elif action_level1 == 1 or action_level1 == 2 or action_level1 == 3: # INTERACT WITH OTHERS (retweets, replies, mentions)
                self.mat_interaction[bot_idx, action_level2] += self.NORMAL_INTEREST
                
                if action_level1 == 1:
                    action = "R"
                elif action_level1 == 2:
                    action = "A"
                else:
                    action = "M"
                self.DNA[self.key.format(i)] += action

            elif action_level1 == 4: # FOLLOW OTHERS
                if not self.mat_followship[bot_idx, action_level2]:
                    self.mat_followship[bot_idx, action_level2] = 1
                    self.mat_interaction[bot_idx, action_level2] += self.FOLLOW_INTEREST
                self.DNA[self.key.format(i)] += "F"

        self.update_retweet()
        self.update_followship()
        self.update_detection()
        reward = self.cal_rewards()

        global_obs = self.pack_observation()

        for i in range(self.n_fake_users):
            key = self.key.format(i)
            if self.done[key] and key not in self.last_obs:
                self.last_obs[key] = global_obs
                self.last_rewards[key] = reward

        if not self.check_all_killed():
            for i in range(self.n_fake_users):
                key = self.key.format(i)
                if not self.done[key]:
                    obs[key], rew[key], done[key], info[key] = global_obs, reward, self.done[self.key.format(i)] != 0, {}
        else:
            obs = self.last_obs
            rew = self.last_rewards
            done = self.done
            # print(rew, self.DNA)

        self.current_t += 1

        return obs, rew, done, info
        


    def close(self):
        self.reset()
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None
