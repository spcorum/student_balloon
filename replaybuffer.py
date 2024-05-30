
# Modified from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from collections import namedtuple, deque
import random
import pickle


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
    