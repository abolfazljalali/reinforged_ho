import gym
import torch
import torch.nn as nn

import pandas as pd

from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from torch.nn import BCELoss


class Monkey(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        nn.Sequential(
            OrderedDict([
                ('layer_1', nn.Linear(in_features, 64)),
                ('activation_1', nn.ReLU()),
                ('layer_2', nn.Linear(64, 64)),
                ('activation_2', nn.ReLU()),
                ('layer_3', nn.Linear(64, 2)),
            ])
        )
        self.softmax = nn.Softmax()

    def forward(self, X):
        y = self.net(X)
        return self.softmax(y)

class MockingBox(Dataset):
    def __init__(self) -> None:
        super(MockingBox, self).__init__()
        self.dataframe = pd.DataFrame(columns=['observation', 'action'])

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        record = self.dataframe.iloc[index]
        obs = record[0]
        action = [0., 1.] if record[1] == 0 else [1., 0.]
        return torch.FloatTensor(obs), torch.FloatTensor(action)

    def store_trajectory(self, trajectory):
        df = pd.DataFrame(trajectory, columns=['observations', 'action'])
        self.dataframe = pd.concat([self.dataframe, df], axis=0)


Monkey()

def gather_trajectories(expert, env, num_games=1, render=False):
    experience = []
    for iteration in range(num_games):
        obs = env.reset()
        done = False
        while not done:
            action = expert.predict(obs, deterministic=True)
            experience.append((obs, action))
            obs, reward, done, info = env.step(action)
            if render:
                env.render()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    expert = PPO.load('expert_10k')
    dataset = MockingBox()
    gather_trajectories(expert, env, num_games=20)