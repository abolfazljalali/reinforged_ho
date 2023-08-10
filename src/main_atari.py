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


class BehaviouralCloningAgent(nn.Module):
    def __init__(self, in_channels, out_features, input_size=(210, 160, 3)):
        super(BehaviouralCloningAgent, self).__init__()
        self.input_size = input_size
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_1 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3)),
            ('maxpool_1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('activ_1', nn.ReLU())]))
        self.conv_2 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)),
            ('maxpool_1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('activ_1', nn.ReLU())]))
        self.conv_3 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)),
            ('maxpool_1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('activ_1', nn.ReLU())]))
        self.flatten = nn.Flatten()
        in_features = self._compute_flatten_feats()
        self.fc = nn.Sequential(OrderedDict([
            ('layer_1', nn.Linear(in_features, 128)),
            ('activation_1', nn.ReLU()),
            ('layer_2', nn.Linear(128, 128)),
            ('activation_2', nn.ReLU()),
            ('layer_3', nn.Linear(128, out_features))
        ]))

        self.softmax = nn.Softmax(dim=1)
        self.optimizer = None
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_1(y)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = self.flatten(y)
        y = self.fc(y)
        return self.softmax(y)

    def _compute_flatten_feats(self):
        x = torch.rand(size=self.input_size)
        with torch.no_grad():
            y = self.avg_pool(x)
            y = self.conv_1(y)
            y = self.conv_2(y)
            y = self.conv_3(y)
            y = self.flatten(y)

        return y.shape[1]

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def predicts(self, x):
        with torch.no_grad():
          y = self.avg_pool(x)
          y = self.conv_1(y)
          y = self.conv_2(y)
          y = self.conv_3(y)
          y = self.flatten(y)
          y = self.fc(y)
          actions_prob = self.softmax(y)

        return torch.argmax(actions_prob)

    def train(self, dataset, num_epochs=1):
        progress_bar = tqdm(range(num_epochs))
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        losses = []
        for e in progress_bar:
            epoch_loss = []
            for obs, action, reward, done in data_loader:
                predicted_action = self.forward(obs)
                loss = self.loss_fn(predicted_action, action)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())
            progress_bar.set_description(f'Epoch {e} - Avg. Loss: {np.sum(epoch_loss) / len(epoch_loss)}')
            losses.append(np.sum(epoch_loss) / len(epoch_loss))

        plot_loss(losses)


class BCDataset(Dataset):
    def __init__(self, images_path, data_path):
        super(BCDataset, self).__init__()
        self.data = pd.read_csv(data_path, sep=',', dtype=str).to_numpy()
        self.names = list(self.data[:, 0])
        self.images_path = images_path
        self.data_path = data_path

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        name = self.names[item]
        img = plt.imread(f'{self.images_path}/{name}.png')[:, :, :3]
        action = np.zeros((4,))
        reward = self.data[item, 2]
        done = self.data[item, 3]
        action[int(self.data[item, 1])] = 1.

        return np.moveaxis(img, -1, 0), action, reward, done


def obs_act_callback(obs_t, obs_tp1, action, rew, done, truncated, info):
    if not isinstance(obs_t, tuple):
        if obs_t.shape == (210, 160, 3):
            timestamp = time.time()
            with open('data.csv', 'a') as f:
                f.write(f'{timestamp},{action},{rew},{done}\n')
            plt.imsave(f'images/{timestamp}.png', obs_t)

def plot_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


# if __name__ == '__main__':
#     # create images folder and initialize data.csv file
#     os.makedirs('images', exist_ok=True)
#     if 'data.csv' not in os.listdir('.'):
#         with open('data.csv', 'a+') as f:
#             f.write('timestamp,action,reward,done\n')
#         f.close()
#     # Uncomment this to play the environment
#     #play(gym.make('ALE/Breakout-v5', render_mode='rgb_array'), zoom=3, callback=obs_act_callback)
#     # Uncomment this for checking data consistency
#     """imgs = os.listdir('images')
#     imgs = [x.rstrip('.png') for x in imgs]
#     df = pd.read_csv('data.csv', sep=',', dtype=str).to_numpy()
#     names = list(df[:, 0])
#     imgs.sort()
#     names.sort()
#     assert set(imgs) == set(names), 'Difference found'"""
#     # Uncomment this for training the network (requires gathered data)
#     dataset = BCDataset(images_path='images', data_path='data.csv')
#     #TODO: Federico's note: Leave these lines in this order, otherwise there will be an error!
    # model = BehaviouralCloningAgent(in_channels=3, out_features=4, input_size=(1, 3, 210, 160))
#     optimizer = Adam(model.parameters(), lr=3e-4)
#     model.set_optimizer(optimizer)
#     model.train(dataset, num_epochs=10)

model = BehaviouralCloningAgent(in_channels=3, out_features=4, input_size=(1, 3, 210, 160))

model.load_state_dict(torch.load('/Users/Aryan/Documents/Repositories/reinforged_ho/src/model_bdst.pt'))
move_data = []
def test_model(obs_t, obs_tp1, action, rew, done, truncated, info):
    if not isinstance(obs_t, tuple):
        if obs_t.shape == (210, 160, 3):
          obs_t = np.moveaxis(obs_t, -1, 0)
          obs_t = obs_t.reshape((1, 3, 210, 160))
          pred = model.predicts(torch.FloatTensor(obs_t))
          move_data.append([action, pred])

# play(gym.make('ALE/Breakout-v5', render_mode='rgb_array'), zoom=3, callback=test_model)
# env.metadata['render_fps'] = 10
# print(move_data)
# plt.plot(move_data)
# plt.show()
# def test_agent(agent, env, num_games=1):
env = gym.make('ALE/Breakout-v5', render_mode='human')
obs, _ = env.reset()
done = False
total_reward = 0.0
env.step(1)
while not done:
    obs = np.moveaxis(obs, -1, 0)
    obs = obs.reshape((1, 3, 210, 160))
    action = model.predicts(torch.FloatTensor(obs))
    print(action.numpy())
    obs, reward, done, _, info = env.step(action.numpy())
    env.render()
    total_reward += reward
#     print(f'Game Total reward: {total_reward}')

# test_agent(model, env, 10)
