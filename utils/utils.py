import numpy as np

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)
        self.index = -1

    def store(self, experience):
        self.index = (self.index + 1) % self.capacity
        self.data[self.index % self.capacity] = experience

    def sample(self, batch_size):
        sample_index = np.random.choice(min(self.index, self.capacity), size=batch_size)
        return self.data[sample_index]

    def return_index(self):
        return self.index