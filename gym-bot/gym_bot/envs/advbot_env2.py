import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from joblib import dump, load
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class AdvBotEnvSpam(gym.Env):
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "R", "A"]
    HISTORY_LENGTH = 20

    def __init__(self, convert_tensor=False):
        self.convert_tensor = convert_tensor
        self.action_encoder = OneHotEncoder(handle_unknown='ignore')
        self.action_encoder.fit(np.array(self.ACTION).reshape(-1,1))

        self.action_space = spaces.Discrete(3)
        self.observation_dim = len(self.convert_state_to_vector(""))
        self.observation_space = spaces.Box(0, 1, shape=(self.observation_dim,), dtype=np.float32)
        self.reset()

        self.N = 10
        self.THRESHOLD = 4
        self.MAX_STEP = 1024


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def convert_state_to_vector(self, state):
        if self.convert_tensor:
            state = list(state)[-self.HISTORY_LENGTH:]
            state = state + ["PAD"]*(self.HISTORY_LENGTH - len(state))
            pad_length = (self.HISTORY_LENGTH - len(state))*len(self.ACTION)
            state = self.action_encoder.transform(np.array(state).reshape(-1,1)).toarray().reshape(1,-1)
            # print(len(state[0]))
            return state[0]
        else:
            return state

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.state += self.ACTION[action]

        done = False
        reward = self.state.count('T')
        count = self.state[-self.N:].count('T')
        if count > self.THRESHOLD:
            done = True

        self.count_steps += 1
        if self.count_steps > self.MAX_STEP:
            done = True

        obs = self.convert_state_to_vector(self.state)
        return obs, reward, done, {}


    def reset(self):
        self.seed()
        self.viewer = None
        self.state = "T"
        self.steps_beyond_done = None
        self.count_steps = 0
        obs = self.convert_state_to_vector(self.state)
        return obs


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
