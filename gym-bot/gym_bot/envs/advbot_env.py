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
        self.vectorizer, self.model = load(model_path) 

    def predict(self, x):
        x = self.vectorizer.transform([x])
        return self.model.predict(x)


class AdvBotEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "R", "A"]
    MAX_STEP = 200
    HISTORY_LENGTH = 100
    INTERVAL = 10

    def __init__(self, convert_tensor=False):
        self.convert_tensor = convert_tensor
        self.action_encoder = OneHotEncoder(handle_unknown='ignore')
        self.action_encoder.fit(np.array(self.ACTION).reshape(-1,1))

        self.action_space = spaces.Discrete(3)
        self.observation_dim = len(self.convert_state_to_vector(""))
        self.observation_space = spaces.Box(0, 1, shape=(self.observation_dim,), dtype=np.float32)

        #load bot detector
        self.model_path = '{}/Documents/gym-advbots/dna/RandomForest_0.87CV.joblib'.format(os.path.expanduser("~"))
        self.detector = Detector(self.model_path)
        print("loaded bot detector", self.detector.model)

        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def convert_state_to_vector(self, state):
        if self.convert_tensor:
            state = list(state)[-self.HISTORY_LENGTH:]
            state = state + ["PAD"]*(self.HISTORY_LENGTH - len(state))
            pad_length = (self.HISTORY_LENGTH - len(state))*len(self.ACTION)
            state = self.action_encoder.transform(np.array(state).reshape(-1,1)).toarray().reshape(1,-1)
            return state[0]
        else:
            return state
            
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.state += self.ACTION[action]
        self.count_steps += 1

        done = False
        reward = 1

        if self.count_steps > self.MAX_STEP:
            done = True

        elif len(self.state) % self.INTERVAL == 0 and len(self.state) > 1:
            pred = self.detector.predict(self.state)[0]
            if pred >= 0.5:
                done = True
                # reward = -1.0
        
        obs = self.convert_state_to_vector(self.state)
        return obs, reward, done, {}


    def reset(self):
        self.seed()
        self.viewer = None
        random_state = np.random.choice(self.ACTION, np.random.choice([1,2,3]))
        self.state = "".join(random_state)
        # self.state = "T"
        self.steps_beyond_done = None
        self.count_steps = 0
        obs = self.convert_state_to_vector(self.state)
        return obs


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
