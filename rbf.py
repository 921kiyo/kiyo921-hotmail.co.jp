import gym
from gym import wrappers
import numpy as np

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
# partial_fit






def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize([value], bins)[0]


class Transformer:
    def __init__(self):
        self.position_bins = np.linspace(-2.4, 2.4, 9)
        self.velocity_bins = np.linspace(-2, 2, 9)
        self.angle_bins = np.linspace(-0.4, 0.4, 9)
        self.velocity_tip_bins = np.linspace(-3.5, 3.5, 9)

    def transform(self, observation):
        position, velocity, angle, velocity_tip = observation
        return build_state([
            to_bin(position, self.position_bins),
            to_bin(velocity, self.velocity_bins),
            to_bin(angle, self.angle_bins),
            to_bin(velocity_tip, self.velocity_tip_bins)
        ])

class Model:
    def __init__(self, env, transformer):
        self.env = env
        self.transformer = transformer
        states = 10**env.observation_space.shape[0]
        actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(states, actions))

    def predict(self, s):
        state = self.transformer.transform(s)
        return self.Q[state]

    def update(self, s, a, G):
        state = self.transformer.transform(s)
        self.Q[state, a] += 1e-2*(G - self.Q[state, a])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            policy = self.predict(s)
            return np.argmax(policy)

def play_one(model, eps, gamma):
    observation = env.reset()
    done = False

    total_reward = 0
    counter = 0

    while not done and counter < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation

        observation, reward, done, _ = env.step(action)

        total_reward += reward

        if done and counter < 199:
            reward = -300

        td_target = reward + gamma*np.max(model.predict(observation))

        model.update(prev_observation, action, td_target)

        counter += 1

    return total_reward

import matplotlib.pyplot as plt

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

if __name__ == "__main__":
    # env = gym.make("CartPole-v0")
    env = gym.make("Pendulum-v0")
    import pdb
    pdb.set_trace()
    env = wrappers.Monitor(env, "record", force=True)

    # if 'monitor' in sys.argv:
    #     filename = os.path.basename(__file__).split('.')[0]
    # monitor_dir = './' + filename + '_' + str(datetime.now())
    # env = wrappers.Monitor(env, monitor_dir)
    tf = Transformer()

    model = Model(env, tf)
    gamma = 0.9

    # dummy call
    target = 0
    input = tf.transform(env.reset(), target=target)

    # model = SGDRegressor()
    # model.partial_fit(input, target)

    N = 10000

    total_rewards = np.empty(N)

    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        total_reward = play_one(model, eps, gamma)
        total_rewards[n] = total_reward
        if n % 100 == 0:
            print("episode:", n, "total reward:", total_reward, "eps:", eps)
    print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
    print("total steps:", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)

