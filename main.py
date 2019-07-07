import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
env = gym.make('CartPole-v1')
env.reset()

episodes = 10000
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
    i = torch.from_numpy(np.random.randn(1, observation_space_size)).float()
    for e in range(episodes):
        state = env.reset()
        episode_reward = 0
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
        #print('Episode reward', episode_reward)
        train_net(net)
    env.close()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(observation_space_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_net(net):
    memory_size = len(replay_memory)
    batch_size = 5
    idxs = np.random.choice(np.arange(memory_size), size=batch_size)
    samples = np.array([[replay_memory[i][j] for j in range(4)] for i in idxs])
    states = samples[:, 0]
    states = np.array([[states[i][j] for j in range(observation_space_size)] for i in range(batch_size)])
    rewards = samples[:, 3].astype('float32')
    rewards = torch.from_numpy(rewards).float()
    next_states = samples[:, 2]
    next_states = np.array([[next_states[i][j] for j in range(observation_space_size)] for i in range(batch_size)])
    actions = samples[:, 1].astype('int32')
    state_q_vals = net.forward(torch.from_numpy(states).float())
    state_q_vals = state_q_vals[range(batch_size), actions]
    next_state_q_vals = net.forward(torch.from_numpy(next_states).float())
    next_state_q_vals, _ = next_state_q_vals.max(1)

    loss = torch.sum(lr * ((rewards + gamma * next_state_q_vals) -  state_q_vals))
    # print("Loss", loss)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


learn()