import gym
from gym import wrappers
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable

from collections import deque

env = gym.make("CartPole-v0")
# env = wrappers.Monitor(env, "ac", force=True)
env.seed(0)

# from reinforce import Policy as PolicyModel
class PolicyModel(nn.Module):
    def __init__(self, s=4, h=16, a=2):
        super().__init__()
        self.fc1 = nn.Linear(s, h)
        self.fc2 = nn.Linear(h, a)

    def forward(self, state):
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        return F.softmax(state, dim=1)

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)


class ValueModel(nn.Module):
    def __init__(self, s=4, h=16, a=2):
        super().__init__()
        self.fc1 = nn.Linear(s, h)
        self.fc2 = nn.Linear(h, a)
        self.fc3 = nn.Linear(a, 1)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        value = self.fc3(state)
        return value


policy = PolicyModel()
critic = ValueModel()
value_model = ValueModel()
actor_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-2)


def actor_critic(n_episodes=1000, max_time=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for episode in range(1, n_episodes + 1):

        log_probs_tensor = []
        values = []
        rewards = []

        masks = []

        state = env.reset()
        for t in range(max_time):
            action, log_prob = policy.choose_action(state)

            value = critic.forward(state)

            log_probs_tensor.append(log_prob)
            values.append(value)

            next_state, reward, done, _ = env.step(action)

            rewards.append(torch.tensor([reward], dtype=torch.float))

            masks.append(torch.tensor([1 - done], dtype=torch.float))

            state = next_state

            if done:
                break

        # After each timestep, update the policy and value

        next_value = critic(next_state)

        # compute returns
        returns = []
        for step in reversed(range(len(rewards))):
            next_value = rewards[step] + gamma * next_value * masks[step]
            returns.insert(0, next_value)
        returns = torch.cat(returns).detach()

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([d * r for d, r in zip(discounts, rewards)])

        values = torch.cat(values)

        advantage = returns - values

        policy_loss = []
        for log_prob in log_probs_tensor:
            policy_loss.append(-log_prob * advantage)

        policy_loss = torch.cat(policy_loss).sum()

        critic_loss = advantage.pow(2).mean()
        critic_loss = Variable(critic_loss, requires_grad=True)

        # Actor (Policy) backprop
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        # Critic (Value model) backprop
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if episode % print_every == 0:
            print(
                "Episode {}\tAverage Score: {:.2f}".format(
                    episode, np.mean(scores_deque)
                )
            )
        if np.mean(scores_deque) >= 195.0:
            print(
                "Environment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    episode - 100, np.mean(scores_deque)
                )
            )
            # break
    return scores
