import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)   # O(1) append/evict

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Double DQN. One network is SHARED by all robots (parameter sharing),
    each robot feeds its own local observation -> works for any number of robots."""

    def __init__(self, input_dim, n_actions, lr=5e-4, gamma=0.95, batch_size=64,
                 buffer_size=50000, warmup=500, target_update=500,
                 eps_start=1.0, eps_final=0.05, eps_decay_steps=15000, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.policy_net = DQNNet(input_dim, n_actions).to(self.device)
        self.target_net = DQNNet(input_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.opt = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_update = target_update
        self.eps_start, self.eps_final, self.eps_decay_steps = eps_start, eps_final, eps_decay_steps
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.act_steps = 0     # drives the exploration schedule
        self.learn_steps = 0   # counts gradient updates (drives target sync)

    def epsilon(self):
        frac = min(1.0, self.act_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_final - self.eps_start)

    def select_action(self, state, greedy=False):
        if not greedy:
            eps = self.epsilon()
            self.act_steps += 1
            if random.random() < eps:
                return random.randrange(self.n_actions)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(s).argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, float(done))

    def train_step(self):
        if len(self.replay_buffer) < max(self.batch_size, self.warmup):
            return None
        s, a, r, s2, d = zip(*self.replay_buffer.sample(self.batch_size))
        s = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.array(s2), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            best = self.policy_net(s2).argmax(dim=1, keepdim=True)      # Double DQN
            next_q = self.target_net(s2).gather(1, best)
            target = r + self.gamma * next_q * (1.0 - d)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.opt.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
