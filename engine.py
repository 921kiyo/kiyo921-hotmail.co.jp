
import matplotlib.pyplot as plt

from reinforce import reinforce
from actor_critic import actor_critic
import numpy as np


# result = actor_critic()
result = reinforce(n_episodes=3000)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(result) + 1), result)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()
