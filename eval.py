import pickle
import gym

from gym import wrappers
from collections import deque
import numpy as np
from actor_critic import PolicyModel, ValueModel
import torch

import matplotlib.pyplot as plt

policy = PolicyModel()
policy.load_state_dict(torch.load("policy"))
policy.eval()
import pdb

# pdb.set_trace()


def evaluate(n_episodes=1000, max_time=1000, print_every=100):
    env = gym.make("CartPole-v0")
    # env = wrappers.Monitor(env, "ac", force=True)
    env.seed(0)

    scores = []
    scores_deque = deque(maxlen=100)
    for episode in range(1, n_episodes + 1):
        rewards = []

        state = env.reset()
        action, _ = policy.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        rewards.append(reward)

        state = next_state

        scores.append(sum(rewards))
        scores_deque.append(sum(rewards))

        if done:
            break

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
    return scores


result = evaluate()

# # pdb.set_trace()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(result) + 1), result)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()
