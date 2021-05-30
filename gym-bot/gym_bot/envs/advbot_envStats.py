import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from joblib import dump, load
import os
import warnings
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
        pred = self.model.predict(x)
        return pred

    def extract_features(self, action, follower=None, following=None):
        num_tweets = action.count('T')
        num_replies = action.count('A')
        num_retweets = action.count('R')
        num_mentions = action.count('M')
    # 
        avg_mentions_per_tweet = num_mentions / max(1, num_tweets)
        retweet_ratio = num_retweets / max(1, num_tweets)
        reply_ratio = num_replies / max(1, num_tweets)
        retweet_reply_ratio = num_retweets / max(1, reply_ratio)
        num_interactions = num_retweets + num_replies + num_mentions
        avg_interaction_per_tweet = num_interactions / max(1, num_tweets)

        rt = [num_tweets, num_replies, num_retweets]
        rt += [retweet_ratio, reply_ratio, retweet_reply_ratio]
        rt += [num_mentions, avg_mentions_per_tweet]
        rt += [num_interactions, avg_interaction_per_tweet]

        # rt = np.array(rt).reshape(1, -1)[:,[9]]
        rt = np.array(rt).reshape(1, -1)
        return rt


class AdvBotEnvStats(gym.Env):
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "R", "A", "M"]
    MAX_STEP = 300
    HISTORY_LENGTH = 50
    INTERVAL = 20

    def __init__(self, convert_tensor=True, seed=777, custom_max_step=None):
        self.seed(seed)
        if custom_max_step:
            self.MAX_STEP = custom_max_step

        self.convert_tensor = convert_tensor
        self.action_encoder = OneHotEncoder(handle_unknown='ignore')
        self.action_encoder.fit(np.array(self.ACTION).reshape(-1,1))

        self.action_space = spaces.Discrete(len(self.ACTION))
        self.observation_dim = len(self.convert_state_to_vector(""))
        self.observation_space = spaces.Box(0, 1, shape=(self.observation_dim,), dtype=np.float32)

        #load bot detector
        self.model_path = '{}/Documents/gym-advbots/traditional/RandomForestClassifier_TRAM_lengthNone.joblib'.format(os.path.expanduser("~"))
        self.detector = Detector(self.model_path)
        print("loaded bot detector", self.detector.model)

        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def convert_state_to_vector(self, current_state):
        if self.convert_tensor:
            state = current_state
            # state = current_state[-self.HISTORY_LENGTH:]
            num_tweets = state.count('T')
            num_replies = state.count('A')
            num_retweets = state.count('R')
            num_mentions = state.count('M')
            obs = [num_tweets, num_replies, num_retweets, num_mentions]
            obs = np.array(obs)
            obs = obs/max(1, obs.sum())
            return obs
        else:
            return current_state
            
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.state += self.ACTION[action]
        self.count_steps += 1

        done = False
        reward = 0.001
              
        if len(self.state) % self.INTERVAL == 0 and len(self.state) > 1:
            pred = self.detector.predict(self.state)[0]
            if pred >= 0.5:
                done = True
            # else:
                # reward += 0.2

        if len(self.state) > self.MAX_STEP:
            done = True
        
        obs = self.convert_state_to_vector(self.state)
        return obs, reward, done, {}

    def render(self, mode="human"):
        print(self.state)

    def reset(self):
        self.seed()
        self.viewer = None
        # self.state = " ".join(np.random.choice(self.ACTION, 12))
        self.state = "T"
        self.steps_beyond_done = None
        self.count_steps = 0
        obs = self.convert_state_to_vector(self.state)
        return obs


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
