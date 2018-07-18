from config import *
from model.mlp import mlp
import tensorflow as tf
import numpy as np
from utils.utils import Memory

class auto_encoder:
    def __init__(
            self,
            learning_rate,
            memory_size,
            batch_size,
            sess,
            output_size
    ):
        self.sess = sess
        self.common_encoder_input = tf.placeholder(tf.float32, shape=[None, n_features], name='common_encoder_input')
        self.common_encoder_output = mlp(inputs=self.common_encoder_input, n_output=output_size, scope='common_encoder_output',
                                    hiddens=[16, 8])
        self.common_decoder_output = mlp(inputs=self.common_encoder_output, n_output=n_features, scope='common_decoder_output')

        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.memory = Memory(self.memory_size)

        self.loss = tf.reduce_mean(tf.squared_difference(self.common_encoder_input, self.common_decoder_output))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def update(self):
        data = self.memory.sample(self.batch_size)

        #not sure if the data is legal

        self.sess.run(self.train, feed_dict = {
            self.common_encoder_input: data
        })

    def store(self, state):
        self.memory.store(np.array([state]))

    @property
    def output(self):
        return self.common_encoder_output

    @property
    def full(self):
        return self.memory.return_index() == self.memory_size-1