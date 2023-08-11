import numpy as np
import matplotlib.pyplot as plt
from receiver import Receiver
import time

# Setup: Qtable and env

low, high = -1, 0# parameters

env = Receiver()  # the enviroment object
env.start()

n_states = [3, 8, 8, 8, 2]
n_actions = 6
actions = range(6)

Q_table = np.random.uniform(low=low, high=high, size=(n_states + [n_actions]))
#Q_table = np.load("qtables/50-qtable.npy")

# Parameters
ALPHA: float = 0.5
GAMMA: float = 0.95
EPISODES: int = 20000
EXPLORATION_BEGIN: int = -200
EXPLORATION_END: int = 1000  # EPISODES//5

LOG_EVERY = 1
SAVE_EVERY = 50
log = []
log_stats = {'ep': [], 'avg': [], 'max': [], 'min': []}

epsilon = .8
EPSILON_DECAY = epsilon / (EXPLORATION_END - EXPLORATION_BEGIN)

time.sleep(1)

env.check_end()
for episode in range(EPISODES):
    state, _ = env.observe()  # starts the env
    print("state", state)
    done = False

    reward_total = 0

    while True:

        if np.random.random() < epsilon:
            # Explore: Get action randomly
            action = np.random.choice(actions)
        else:
            # Exploit: Get action from Q_table
            action = np.argmax(Q_table[state])


        env.act(action)
        done, reward = env.check_end()

        if done:
            sa = state + (action,)
            Q_table[sa] = reward
            reward_total += reward
            break

        state_new, reward = env.observe()
        if reward == 50:
            print(reward)
        reward_total += reward

        # \max_a Q(s_{t+1}, a)
        try:
            Q_next = np.max(Q_table[state_new])
        except IndexError as e:
            print(state_new)
            raise e

        # Q(s_t, a_t)
        Q_current = Q_table[state + (action,)]

        # Q^{new}(s_t, a_t)
        Q_new = (1 - ALPHA) * Q_current + ALPHA * (reward + GAMMA * Q_next)

        # Update the Q table
        Q_table[state + (action,)] = Q_new

        state = state_new

    if EXPLORATION_END >= episode >= EXPLORATION_BEGIN:
        epsilon -= EPSILON_DECAY
        epsilon = max(epsilon, 0)

    log.append(reward_total)
    if not episode % LOG_EVERY:
        log_stats['ep'].append(episode)
        log_stats['avg'].append(sum(log) / len(log))
        log_stats['max'].append(max(log))
        log_stats['min'].append(min(log))
        print(f'Episode: {episode:>5d}, average reward: {sum(log) / len(log):>4.1f}, current epsilon: {epsilon:>1.2f}')
        log = []
    if not episode % SAVE_EVERY:
        np.save(f"qtables/{episode}-qtable.npy", Q_table)


plt.plot(log_stats['ep'], log_stats['avg'], label="average rewards")
plt.plot(log_stats['ep'], log_stats['max'], label="max rewards")
plt.plot(log_stats['ep'], log_stats['min'], label="min rewards")
plt.legend(loc=4)
plt.show()