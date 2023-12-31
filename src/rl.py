import gym
import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from collections import OrderedDict

from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from torch.nn import BCELoss
from torch.optim import Adam

import matplotlib.pyplot as plt
from tqdm import tqdm

class Monkey(nn.Module):
    def __init__(self, in_features, out_feature) -> None:
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict([
                ('layer_1', nn.Linear(in_features, 64)),
                ('activation_1', nn.ReLU()),
                ('layer_2', nn.Linear(64, 64)),
                ('activation_2', nn.ReLU()),
                ('layer_3', nn.Linear(64, out_feature)),
            ])
        )
        self.softmax = nn.Softmax()
        self.optimizer = Adam(self.net.parameters(), lr=3e-4)
        self.loss_fn = BCELoss()

    def forward(self, X):
        y = self.net(X)
        return self.softmax(y)

    def predict(self, obs):
        with torch.no_grad():
            y = self.net(obs)
            action_prob = self.softmax(y)
            return torch.argmax(action_prob)
    
    def train(self, dataset, num_epochs=1):
        progress_bar = tqdm(range(num_epochs))
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        losses = []
        for e in progress_bar:
            epoch_loss = []
            for obs, action in data_loader:
                predicted_actions = self.forward(obs)
                loss = self.loss_fn(predicted_actions, action)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss.append(loss.item())
            progress_bar.set_description(f'Epoch {e} Average Loss: {np.mean(epoch_loss)}')
            losses.append(np.mean(epoch_loss))


class MockingBox(Dataset):
    def __init__(self) -> None:
        super(MockingBox, self).__init__()
        self.dataframe = pd.DataFrame(columns=['observation', 'action'])

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        record = self.dataframe.to_numpy()[index]
        obs = record[0]
        action = [0., 1.] if record[1] == 0 else [1., 0.]
        print(obs, action)
        return torch.FloatTensor(obs), torch.FloatTensor(action)

    def store_trajectory(self, trajectory):
        df = pd.DataFrame(trajectory, columns=['observations', 'action'])
        self.dataframe = pd.concat([self.dataframe, df], axis=0)


def gather_trajectories(expert, env, dataset, num_games=1, render=False):
    experience = []
    for iteration in range(num_games):
        obs,_ = env.reset()
        done = False
        while not done:
            action, _ = expert.predict(obs, deterministic=True)
            experience.append((obs, action))
            obs, reward, done, _, info = env.step(action)
            if render:
                env.render()
        dataset.store_trajectory(experience)

def agent_trajectories(agent, env, num_games=1):
    for iteration in range(num_games):
        obs,_ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.predict(torch.FloatTensor(obs))
            obs, reward, done, _, info = env.step(action.numpy())
            total_reward += reward
            env.render()

        print(f'Total reward: {total_reward}')

def plot_losses(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    expert = PPO.load('model/expert_10k')
    agent = Monkey(in_features=4, out_feature=2)
    dataset = MockingBox()
    gather_trajectories(expert, env, dataset, num_games=20)
    agent_trajectories(agent, env, 10)
    agent.train(dataset, 25)
    agent_trajectories(agent, env, 10)
