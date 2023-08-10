import os

import gym
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import OrderedDict
from tqdm import tqdm
from gym.utils.play import play

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss



def obs_act_callback(obs_t, obs_tp1, action, rew, done, truncated, info):
    if not isinstance(obs_t, tuple):
        if obs_t.shape == (210, 160, 3):
            timestamp = time.time()
            with open('data.csv', 'a') as f:
                f.write(f'{timestamp},{action},{rew},{done}\n')
            plt.imsave(f'images/{timestamp}.png', obs_t)

if __name__ == '__main__':
    # create images folder and initialize data.csv file
    os.makedirs('images', exist_ok=True)
    if 'data.csv' not in os.listdir('.'):
        with open('data.csv', 'a+') as f:
            f.write('timestamp,action,reward,done\n')
        f.close()
    play(gym.make('ALE/Breakout-v5', render_mode='rgb_array'), zoom=3, callback=obs_act_callback)