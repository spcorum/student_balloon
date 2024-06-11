
# Modified from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from collections import namedtuple, deque
import random
import pickle
import numpy as np
import torch


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):

    def __init__(self,
                 capacity: int,
                 fields = ('state', 'action', 'next_state', 'reward')
                 ):
        self.Transition = namedtuple('Transition', fields)
        self.fields = fields
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = self.Transition(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.memory)
    
    def save(self, path):
        return
        with open(path, 'w') as f:
            pickle.dump(dict(fields=self.fields, memory=self.memory), f)

    def load(self, path):
        return
        with open(path, 'r') as f:
            d = pickle.load(f)
        self.Transition = namedtuple('Transition', *d['fields'])
        self.fields = d['fields']
        self.memory = d['memory']




# Thank you ChatGPT :)
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def add(self, transition, td_error):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        batch = list(zip(*samples))
        states = torch.tensor(batch[0], dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(batch[3], dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32)
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error)

# Usage example
buffer = PrioritizedReplayBuffer(capacity=10000)
# Add transitions to the buffer
# buffer.add((state, action, reward, next_state, done), td_error)
# Sample a batch of transitions
# states, actions, rewards, next_states, dones, weights, indices = buffer.sample(batch_size=32)
