import gym
import os
import random
import sys
import time

import ray
import glob
from tqdm import tqdm
from ge.graph_utils import *

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

