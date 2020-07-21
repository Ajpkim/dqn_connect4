from collections import namedtuple
import random

Experience = namedtuple('Experience', field_names=('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, seed=3):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.seed = random.seed(seed)
    
    def push(self, state, action, reward, new_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # avoid index errors below
        self.memory[self.position] = Experience(state, action, reward, new_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    pass