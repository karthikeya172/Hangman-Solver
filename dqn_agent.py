# dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'avail_mask'))

# ---------- Prioritized Replay ----------
class PrioritizedReplayBuffer:
    def __init__(self, capacity=80000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, td_error=None):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if td_error is not None:
            max_prio = abs(td_error) + 1e-5
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-5


# ---------- Dueling Network ----------
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        s = self.shared(x)
        v = self.value(s)
        a = self.adv(s)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


# ---------- Agent with PER ----------
class DQNAgent:
    def __init__(self, state_dim, action_dim=26, lr=5e-5, gamma=0.99,
                 buffer_size=80000, batch=128, device=None,
                 alpha=0.6, beta_start=0.4, beta_frames=4000):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch = batch
        self.tau = 0.01
        self.steps = 0
        self.min_replay = 2000

        self.net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-6)

        self.replay = PrioritizedReplayBuffer(capacity=buffer_size, alpha=alpha)
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / max(1, self.beta_frames))

    def select_action(self, state, avail_mask, epsilon):
        # epsilon-greedy with HMM-guided exploration
        if random.random() < epsilon:
            avail = np.where(avail_mask > 0.5)[0]
            if len(avail) == 0:
                return random.randrange(self.action_dim)
            try:
                hmm_vec = state[-26:]
                probs = np.clip(hmm_vec, 1e-6, None)
                probs = probs * avail_mask
                s = probs.sum()
                if s > 0:
                    probs = probs / s
                    return int(np.random.choice(np.arange(self.action_dim), p=probs))
            except Exception:
                return int(random.choice(avail))
            return int(random.choice(avail))

        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.net(st).cpu().numpy()[0]
            q = q - 1e6 * (1.0 - avail_mask)
            return int(np.argmax(q))

    def store(self, *args):
        self.replay.add(Transition(*args))

    def train_step(self):
        if len(self.replay) < self.min_replay:
            return None

        beta = self.beta_by_frame(self.steps)
        samples, indices, weights = self.replay.sample(self.batch, beta)
        batch = Transition(*zip(*samples))

        states = torch.tensor(np.vstack(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_avail_masks = torch.tensor(np.vstack(batch.avail_mask), dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals = self.net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target(next_states)
            penalized_next_q = next_q - (1.0 - next_avail_masks) * 1e6
            max_next_q, _ = penalized_next_q.max(dim=1, keepdim=True)
            target = rewards + self.gamma * (1.0 - dones) * max_next_q

        td_errors = (target - q_vals).detach().cpu().numpy().flatten()
        # L1 (absolute) TD loss weighted by importance-sampling weights
        loss = (weights * torch.abs(target - q_vals)).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()

        self.replay.update_priorities(indices, td_errors)

        # soft update
        for param, target_param in zip(self.net.parameters(), self.target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.steps += 1
        return loss.item()
