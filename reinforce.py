import gym
from gym import wrappers
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from collections import deque

env = gym.make("CartPole-v0")
env = wrappers.Monitor(env, "reinforce", force=True)
env.seed(0)

from actor_critic import PolicyModel as Policy

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def reinforce(n_episodes=1000, max_time=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for episode in range(1, n_episodes + 1):

        log_probs_tensor = []
        rewards = []

        state = env.reset()
        for t in range(max_time):
            action, log_prob = policy.choose_action(state)

            log_probs_tensor.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([d * r for d, r in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in log_probs_tensor:
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

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


