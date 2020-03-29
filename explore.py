# Simple data exploration by random actions.

import gym

env = gym.make("CartPole-v0")
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)  # take a random action
    print("obs: ", observation, "action: ", action, "reward: ", reward, "done: ", done)
env.close()
