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
        return self.model.predict(x)

    def extract_features(self, action, follower=None, following=None):
        num_tweets = action.count('T')
        num_interactions = action.count('I')
        avg = num_interactions / max(1, num_tweets)
        return np.array([avg]).reshape(1, -1)


class AdvBotEnvStatsTI(gym.Env): #advbotStats-v1
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "I"]
    MAX_STEP = 30
    TRESHOLD = 10
    HISTORY_LENGTH = 20

    def __init__(self, convert_tensor=False, seed=777):
        self.seed(seed)
        self.convert_tensor = convert_tensor
        self.action_encoder = OneHotEncoder(handle_unknown='ignore')
        self.action_encoder.fit(np.array(self.ACTION).reshape(-1,1))

        self.action_space = spaces.Discrete(len(self.ACTION))
        self.observation_dim = len(self.convert_state_to_vector(""))
        self.observation_space = spaces.Box(0, 1, shape=(self.observation_dim,), dtype=np.float32)

        #load bot detector
        self.model_path = '{}/Documents/gym-advbots/traditional/RandomForestClassifier_stats_lengthNone.joblib'.format(os.path.expanduser("~"))
        self.detector = Detector(self.model_path)
        print("loaded bot detector", self.detector.model)

        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def convert_state_to_vector(self, state):
        if self.convert_tensor:
            num_tweets = state.count('T')
            num_interactions = state.count('I')
            obs = [num_tweets, num_interactions]
            obs = np.array(obs)
            obs = obs/self.MAX_STEP
            return obs
        else:
            return state
            
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.state += self.ACTION[action]
        self.count_steps += 1

        done = False
        reward = 0
        if self.count_steps % self.TRESHOLD == 0 and self.count_steps > 1:
            pred = self.detector.predict(self.state)[0]
            if pred >= 0.5:
                done = True
            else:
                reward = 10

        if self.count_steps > self.MAX_STEP:
            done = True
        
        obs = self.convert_state_to_vector(self.state)
        return obs, reward, done, {}

    def render(self, mode="human"):
        print(self.state)

    def reset(self):
        self.seed()
        self.viewer = None
        self.state = ""
        self.steps_beyond_done = None
        self.count_steps = 0
        obs = self.convert_state_to_vector(self.state)
        return obs

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
