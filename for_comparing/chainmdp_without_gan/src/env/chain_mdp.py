import sys
import time
import numpy as np


class ChainMDP:

    def __init__(self, length):
        self.length = length
        self.loc = 0

    def step(self, action):
        action = 1 if action == 1 else -1
        self.loc = self.loc + action
        if self.loc < 0:
            self.loc += 1
            return 0.1, self.get_state()
        elif self.loc == self.length:
            self.loc -= 1
            return 1, self.get_state()
        else:
            return 0, self.get_state()

    def render(self, fps=50):
        image = '-' * self.loc + '*' + '-' * (self.length - self.loc - 1)
        sys.stdout.write(image)
        sys.stdout.flush()
        print('\033[1A')  # printer go back to left
        time.sleep(1.0 / fps)

    def get_state(self):
        return (np.arange(self.length) == self.loc).astype(np.integer)

    def reset(self):
        self.loc = 0
        return self.get_state()


if __name__ == '__main__':
    env = ChainMDP(10)
    for i in range(100):
        r, s = env.step(-1)
        env.render(10)
