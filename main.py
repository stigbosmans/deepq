import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
env = gym.make('CartPole-v1')
env.reset()

episodes = 10
max_steps = 200

gamma = 0.99
lr = 0.1
epsilon = 0.99
epsilon_decay = 0.005
epsilon_min = 0.05
epsilon_max = 1.0

batch_size = 24
action_space_size = env.action_space.n
observation_space_size = env.observation_space.shape[0]
episode_rewards = []
replay_memory = deque(maxlen=10000)


def learn():
    global epsilon
    net = Net()
    print(np.array([1,2,3,4]).shape)
    i = torch.from_numpy(np.random.randn(1, observation_space_size)).float()
    episode_reward = 0
    for e in range(episodes):
        state = env.reset()
        for s in range(max_steps):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                x = torch.from_numpy(np.array([state])).float()
                with torch.no_grad():
                    action = np.argmax(net.forward(x).numpy())
            next_state, reward, done, info = env.step(action)
            replay_memory.append((state, action, next_state, reward))
            if done:
                break
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-epsilon_decay * e)
            episode_reward += reward
            state = next_state
        print('finished', s)
    env.close()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(observation_space_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_net(net):
    memory_size = len(replay_memory)
    idxs = np.random.choice(np.arange(memory_size), size=memory_size)
    samples = np.array([replay_memory[i] for i in idxs])
    states = samples[:, 0]
    actions = net.forward(torch.from_numpy(states).float())


learn()