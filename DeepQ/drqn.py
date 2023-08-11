import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import pickle
import random
import torch
import torch.nn as nn
from receiver import Receiver

INPUT_IMAGE_DIM = 84
OUT_SIZE = 4
BATCH_SIZE = 32
TIME_STEP = 8
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
TOTAL_EPISODES = 20000
MAX_STEPS = 50
MEMORY_SIZE = 3000
UPDATE_FREQ = 5
PERFORMANCE_SAVE_INTERVAL = 500
TARGET_UPDATE_FREQ = 20000 #step

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class RecurrentReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size, time_step=6):
        sampled_episodes = random.sample(self.memory, batch_size)
        batch = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - time_step)
            batch.append(episode[point:point + time_step])

        return batch

    def __len__(self):
        return len(self.memory)


class RNN(nn.Module):

    def __init__(self, device, input_size, out_size, hidden_size=10):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.hidden_size = 10
        self.device = device

        self.lstm_layer = nn.RNN(input_size, hidden_size, 1)
        self.out = nn.Linear(in_features=hidden_size, out_features=out_size)

    def forward(self, x, h=None):
        h = torch.zeros(self.hidden_size).float().to(self.device)

        # conexões da rede usando a camada ridden e
        y, hidden = self.lstm_layer(x, h)

        # e as conexões com o layer de saída (linear)
        y = self.out(y)

        return y, hidden.detach()

env = Receiver()
env.start()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
memory = RecurrentReplayMemory(MEMORY_SIZE)

policy = RNN(device, input_size=8, out_size=6).float().to(device)
target = RNN(device, input_size=8, out_size=6).float().to(device)
print(policy)

target.load_state_dict(policy.state_dict())
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(policy.parameters(), lr=0.00025)

# Fill memory
for i in range(0, 100):

    prev_state, _ = env.observe()
    print(prev_state)
    step_count = 0
    local_memory = []
    done = False

    while not done:
        step_count += 1
        action = np.random.randint(0, 6)
        env.act(action)
        done, reward = env.check_end()
        if not done:
            next_state, reward = env.observe()

            local_memory.append((prev_state, action, next_state, reward))

        prev_state = next_state

    print(local_memory)
    for i in local_memory:
        memory.push(*i)

print('Populated with %d Episodes' % (len(memory)))

# Start Algorithm
epsilon = INITIAL_EPSILON
loss_stat = []
reward_stat = []
total_steps = 0

for episode in range(TOTAL_EPISODES):

    prev_state, _ = env.observe()
    total_reward = 0
    step_count = 0
    local_memory = []

    hidden_state = None
    done = False

    while not done:

        step_count += 1
        total_steps += 1

        if np.random.rand(1) < epsilon:
            torch_x = torch.from_numpy(prev_state).to(device)
            _, hidden_state = policy.forward(torch_x, hidden_state)
            action = np.random.randint(0, 4)

        else:
            torch_x = torch.from_numpy(prev_state).float().to(device)
            action, hidden_state = policy.forward(torch_x, hidden_state)
            action = int(torch.argmax(action[0]))

        env.act(action)
        done, reward = env.check_end()
        if not done:
            next_state, reward = env.observe()
        else:
            next_state = np.zeros(8)
        total_reward += reward

        local_memory.append((prev_state, action, reward, next_state))

        prev_state = next_state

        if (total_steps % TARGET_UPDATE_FREQ) == 0:
            target.load_state_dict(policy.state_dict())

        if (total_steps % UPDATE_FREQ) == 0:

            hidden_batch = None

            batch = memory.sample(batch_size=BATCH_SIZE, time_step=TIME_STEP)

            current_states = []
            acts = []
            rewards = []
            next_states = []

            for b in batch:
                cs, ac, rw, ns = [], [], [], []
                for element in b:
                    cs.append(element[0])
                    ac.append(element[1])
                    rw.append(element[2])
                    ns.append(element[3])
                current_states.append(cs)
                acts.append(ac)
                rewards.append(rw)
                next_states.append(ns)

            current_states = np.array(current_states)
            acts = np.array(acts)
            rewards = np.array(rewards)
            next_states = np.array(next_states)

            torch_current_states = torch.from_numpy(current_states).float().to(device)
            torch_acts = torch.from_numpy(acts).long().to(device)
            torch_rewards = torch.from_numpy(rewards).float().to(device)
            torch_next_states = torch.from_numpy(next_states).float().to(device)

            Q_next, _ = target.forward(torch_next_states, hidden_state)
            Q_next_max, __ = Q_next.detach().max(dim=1)
            target_values = torch_rewards[:, TIME_STEP - 1] + (GAMMA * Q_next_max)

            Q_s, _ = policy.forward(torch_current_states, hidden_state)
            Q_s_a = Q_s.gather(dim=1, index=torch_acts[:, TIME_STEP - 1].unsqueeze(dim=1)).squeeze(dim=1)

            loss = criterion(Q_s_a, target_values)

            #  save performance measure
            loss_stat.append(loss.item())

            # make previous grad zero
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update params
            optimizer.step()

    # save performance measure
    reward_stat.append(total_reward)

    memory.push(Transition(prev_state, action, next_state, reward))

    if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / TOTAL_EPISODES

    if (episode + 1) % PERFORMANCE_SAVE_INTERVAL == 0:
        perf = {}
        perf['loss'] = loss_stat
        perf['total_reward'] = reward_stat

    # print('Episode : ',episode+1,'Epsilon : ',epsilon,'Reward : ',total_reward,)

