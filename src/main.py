import gym
import sys
import argparse

from stable_baselines3 import DQN, PPO


def train_DQN(env, policy, total_steps, learning_starts):
    model = DQN(policy=policy, env=env, learning_starts=learning_starts)
    model.learn(total_timesteps=total_steps)
    model.save('model/dqn_wtf')

def train_PPO(env, policy, total_steps, learning_starts):
    model = PPO(policy=policy, env=env, learning_starts=learning_starts)
    model.learn(total_timesteps=total_steps)
    model.save('model/ppo_wtf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='A program for Reinforcement Learning fun!',
        prog='Reinforge!'
    )

    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-p', '--test', action='store_true')
    parser.add_argument('-m', '--model', default='dqn')
    args = parser.parse_args()

    env = gym.make('MountainCar-v0', render_mode='human')
    model = DQN.load('model/dqn_wtf')
    # model.learn(total_timesteps=50000)
    # model.save('model/dqn_wtf')
    num_games = 20
    # if args['train']:
    #     if args['model'] == 'dqn':
    #         train_DQN(env, 'MlpPolicy', 50000, 1)
    #     if args['model'] == 'ppo':
    #         train_PPO(env, 'MlpPolicy', 50000, 1)
    # if args['model'] == 'dqn':
    #     model = DQN.load('model/dqn_wtf')

    # if args['model'] == 'ppo':
    #     model = DQN.load('model/ppo_wtf')

    for i in range(num_games):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, _, info = env.step(action=action)
            env.render()
