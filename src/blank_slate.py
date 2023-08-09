import torch
import torch.nn as nn
import gym
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from tqdm import tqdm

from torch.nn import BCELoss
from torch.optim import Adam


class BehaviouralCloningAgent(nn.Module):
    def __init__(self, in_features, out_features):
        super(BehaviouralCloningAgent, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('layer_1', nn.Linear(in_features, 64)),
            ('activation_1', nn.ReLU()),
            ('layer_2', nn.Linear(64, 64)),
            ('activation_2', nn.ReLU()),
            ('layer_3', nn.Linear(64, out_features))
        ]))

        self.softmax = nn.Softmax()
        self.optimizer = Adam(self.net.parameters(), lr=3e-4)
        self.loss_fn = BCELoss()

    def forward(self, x):
        y = self.net(x)
        return self.softmax(y)

    def predict(self, obs):
        with torch.no_grad():
            y = self.net(obs)
            actions_prob = self.softmax(y)

        return torch.argmax(actions_prob)

    def train(self, dataset, num_epochs=1):
        progress_bar = tqdm(range(num_epochs))
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        losses = []
        for e in progress_bar:
            epoch_loss = []
            for obs, action in data_loader:
                predicted_action = self.forward(obs)
                loss = self.loss_fn(predicted_action, action)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())
            progress_bar.set_description(f'Epoch {e} - Avg. Loss: {np.sum(epoch_loss) / len(epoch_loss)}')
            losses.append(np.sum(epoch_loss) / len(epoch_loss))

        plot_loss(losses)


class ObservationActionDataset(Dataset):
    def __init__(self):
        super(ObservationActionDataset, self).__init__()
        self.dataframe = pd.DataFrame(columns=['observation', 'action'])

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, item: int):
        row = self.dataframe.to_numpy()[item, :]
        obs = row[0]
        action = row[1]
        action = [1., 0.] if action == 0 else [0., 1.]

        return torch.FloatTensor(obs), torch.FloatTensor(action)

    def store_trajectory(self, trajectory):
        df = pd.DataFrame(trajectory, columns=['observation', 'action'])
        self.dataframe = pd.concat([self.dataframe, df], axis=0)


def gather_trajectories(expert, env, dataset, num_games=1, render=False):
    experience = []
    for i in range(num_games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = expert.predict(obs, deterministic=True)
            experience.append((obs, action))
            obs, reward, done, _, info = env.step(action)
            if render:
                env.render()

    dataset.store_trajectory(experience)


def test_agent(agent, env, num_games=1):
    for i in range(num_games):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.predict(torch.FloatTensor(obs))
            obs, reward, done, _, info = env.step(action.numpy())
            env.render()
            total_reward += reward
        print(f'Game {i} - Total reward: {total_reward}')


def plot_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    expert = PPO.load("model/expert_10k")
    dataset = ObservationActionDataset()
    agent = BehaviouralCloningAgent(in_features=4, out_features=2)

    gather_trajectories(expert, env, dataset, num_games=20)

    test_agent(agent, env, num_games=10)

    agent.train(dataset, num_epochs=15)

    test_agent(agent, env, num_games=10)
