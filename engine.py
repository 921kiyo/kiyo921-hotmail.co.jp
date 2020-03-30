
import matplotlib.pyplot as plt

from reinforce import reinforce
from actor_critic import actor_critic
import numpy as np


results = []
for i in range(10):
    result = actor_critic(n_episodes=10000)
    # result = reinforce(n_episodes=10000)
    results.append(result)

avg = [float(sum(l))/len(l) for l in zip(*results)]
# avg = result
import pdb
# pdb.set_trace()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(avg) + 1), avg)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()
