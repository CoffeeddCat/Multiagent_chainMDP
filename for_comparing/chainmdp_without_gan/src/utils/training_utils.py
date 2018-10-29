from src.constants.config import *

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)
        self.index = -1

    def store(self, experience):
        self.index += 1
        self.data[self.index % self.capacity] = experience

    def sample(self, batch_size):
        sample_index = np.random.choice(min(self.index, self.capacity), size=batch_size)
        return self.data[sample_index]


class History:
    def __init__(self, length):
        self.history = np.zeros(length, dtype=object)
        self.length = length
        self.index = 0
        self.total = 0

    def put(self, data):
        self.history[self.index] = data
        self.index = (self.index + 1) % self.length
        self.total += 1

    def get(self):
        if not self.full():
            print('Warning : fetch data from uncompleted history')
            input('Press any button to continue')
        return np.append(self.history[self.index:], self.history[:self.index])

    def full(self):
        return self.total >= self.length

    def clear(self):
        self.total = 0
        self.index = 0


class ScorePlotter:
    def __init__(self):
        self.recent_twenty = []
        self.score_history = []
        plt.ion()
        self.fig = plt.figure()
        self.x_upperlim = 20
        self.ax = plt.axes(xlim=(0, self.x_upperlim), ylim=(-10, 120))
        self.ax.grid()
        self.line, = self.ax.plot([], [], lw=2)
        plt.pause(0.02)

    @property
    def plotter_animate(self):
        self.line.set_data(range(len(self.score_history)), self.score_history)
        if len(self.score_history) > self.x_upperlim:
            self.x_upperlim += 20
            self.ax.set_xlim(0, self.x_upperlim)
            self.ax.figure.canvas.draw()
        return self.line,

    def plot(self, score):
        self.recent_twenty.append(score)
        if len(self.recent_twenty) >= 50:
            self.score_history.append(np.mean(self.recent_twenty))
            animation.FuncAnimation(self.fig, self.plotter_animate, blit=False)
            del self.recent_twenty[0]
            plt.pause(0.02)


def plot_score(score_plotter, score_queue):
    if not score_queue.empty():
        score = score_queue.get()
        score_plotter.plot(score)


def synchronize_version(local_ai, global_ai):
    if local_ai.learn_step != global_ai.learn_step:
        local_ai.sync()


def fetch_data(ai, gan, data_queue):
    """Every 200 frames train one time."""
    if not data_queue.empty():
        # for i in range(200):
        while not data_queue.empty():
            exp = data_queue.get()
            if USING_GAN:
                gan.store(exp[0][0])
            ai.store(exp)
