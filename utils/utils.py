import numpy as np
from config import *

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

def trans_to_one_hot(input):
    input = np.array(input) + 1
    output = []
    for i in range(ai_number):
        for j in range(chain_length+1):
            if j==input[i]:
                output.append(1)
            else:
                output.append(0)
    return output