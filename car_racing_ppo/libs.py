import torch as t
import random
import torch.nn as nn
from collections import deque
import numpy as np
import time
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden1, hidden2):
        super(ActorCritic, self).__init__()

        shape = input_dim[0]
        channel_size = input_dim[2]
        pool_size_1 = 4
        pool_size_2 = 4
        total_pooling = pool_size_1 * pool_size_2
        self.out1 = 12
        self.out2 = 12
        self.out3 = 12
        self.fcn_input_dim = shape ** 2 / total_pooling ** 2 * self.out3
        self.fcn_input_dim = int(self.fcn_input_dim)

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=self.out1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pool_size_1),
            nn.Conv2d(in_channels=self.out1, out_channels=self.out2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pool_size_2),
            nn.Conv2d(in_channels=self.out2, out_channels=self.out3, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.mu_net = nn.Sequential(
            nn.Linear(self.fcn_input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
            nn.Tanh()
        )

        self.var_net = nn.Sequential(
            nn.Linear(self.fcn_input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
            nn.Softplus()
        )

        self.critic_net = nn.Sequential(
            nn.Linear(self.fcn_input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        conv_out = self.conv_net(x)
        if conv_out.shape == t.Size([self.out3, 6, 6]):
            conv_out = t.flatten(conv_out, start_dim=0)
        else:
            conv_out = t.flatten(conv_out, start_dim=1)
        mu = self.mu_net(conv_out)
        variance = self.var_net(conv_out)
        dists = t.distributions.Normal(mu, variance ** (1 / 2))
        values = self.critic_net(conv_out)

        return dists, values


class ReplayMemory:
    def __init__(self, batch_size, max_len):
        self.batch_size = batch_size
        self.max_len = max_len
        self.memory = deque(maxlen=self.max_len)

    def __len__(self):
        return len(self.memory)

    def add_data(self, state, action, log_prob, reward, value, done):
        self.memory.append((state, action, log_prob, reward, value, done))

    def get_memory(self):
        arr = np.array(self.memory, dtype=object)
        states = arr[:, 0]
        actions = arr[:, 1]
        log_probs = arr[:, 2]
        rewards = arr[:, 3]
        values = arr[:, 4]
        dones = arr[:, 5]
        return states, np.vstack(actions), np.vstack(log_probs), np.vstack(rewards), np.vstack(values), np.vstack(dones)

    def generate_batches(self, states, actions, log_probs, returns, values):
        indexes = list(np.arange(len(states)))
        random.shuffle(indexes)
        batch_indexes = np.array_split(indexes, self.max_len // self.batch_size)

        batches = []
        for batch_index in batch_indexes:
            batch = []
            for index in batch_index:
                batch.append((states[index], actions[index], log_probs[index], returns[index], values[index]))
            batches.append(batch)
        return batches


class Updater:
    def __init__(self, net, clip_param, critic_discount, entropy_beta, learning_rate=0.001):
        self.net = net
        self.clip_param = clip_param
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta
        self.optimizer = t.optim.Adam(self.net.parameters(), lr=learning_rate)

    def update(self, states, actions, old_log_probs, returns, advantage):
        dist, new_value = self.net(states)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)
        ratio = new_log_probs.exp() / old_log_probs.exp()

        surrogate1 = ratio * advantage
        surrogate2 = t.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

        actor_loss = -t.min(surrogate1, surrogate2).mean()
        critic_loss = (returns - new_value).pow(2).mean()

        total_loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


def gae_calculator(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - int(masks[step])) - values[step]
        gae = delta + gamma * lam * (1 - int(masks[step])) * gae
        returns.insert(0, gae + values[step])
    return returns


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def visualize(data):
    if type(data) == type("string"):
        file_path = data
        file = open(file_path, "r")
        lines = file.readlines()
        plotting_indexes = range(0, len(lines), 10)

        rewards = []
        avg_rewards = []
        for line in lines:
            reward = line.split(" ")[1][:-1]
            rewards.append(float(reward))
        for index, reward in enumerate(rewards):
            if len(rewards) - index > 10:
                avg_10_reward = np.array(rewards[index:index + 10], dtype=float).mean()
                avg_rewards.append(avg_10_reward)

        rewards_arr = np.array(rewards, dtype=float)
        avg_arr = np.array(avg_rewards, dtype=float)
        plt.plot(plotting_indexes, rewards_arr[plotting_indexes])
        plt.plot(plotting_indexes[:-1], avg_arr[plotting_indexes[:-1]], c="red")
        plt.legend(["epoch_reward", "10_epochs_average_reward"])

    else:
        plt.plot(range(len(data)), data, c="red")
        plt.legend(["epoch_reward"])

    plt.savefig(f"reward_by_epoch_{int(time.time())}.png")
    plt.show()
