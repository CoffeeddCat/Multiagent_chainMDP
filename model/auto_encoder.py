#This encoder is without grad from the encoder.
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

        #state_t
        self.encoder_input = tf.placeholder(tf.float32, shape=[None, n_features], name='encoder_input')
        self.encoder_output = mlp(inputs=self.encoder_input, n_output=output_size, scope='encoder_output',
                                    hiddens=[32, 16, 8])
        self.decoder_output = mlp(inputs=self.encoder_output, n_output=n_features, scope='decoder_output', hiddens=[8, 16, 32])
        self.encoder_output_ = tf.stop_gradient(self.decoder_output)

        #some const
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size

        #memory
        self.memory = Memory(self.memory_size)

        #for train
        self.loss = tf.reduce_mean(tf.squared_difference(self.encoder_input, self.decoder_output))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
 
    def learn(self):
        data = self.memory.sample(self.batch_size)

        state = []

        for i in range(self.batch_size):
            state.append(data[i][0])

        self.sess.run(self.train, feed_dict = {
            self.encoder_input: state
        })

    def store(self, state):
        self.memory.store(np.array([state]))

    def output(self, state):

        return self.sess.run(self.encoder_output, feed_dict = {
            self.encoder_input: [np.array(state)]
            })

    def output_loss(self):
        data = self.memory.sample(self.batch_size)

        state = []

        for i in range(self.batch_size):
            state.append(data[i][0])

        temp = self.sess.run(self.loss, feed_dict = {
            self.encoder_input: state
        })
        print('now loss:', temp)

    @property
    def full(self):
        return self.memory.return_index() == self.memory_size-1

    def process_data(self):
        state, action, reward, state_next, done, decays = [], [], [], [], [], []
        temp = self.memory.sample(self.batch_size)
        for i in range(self.batch_size):
            state.append(temp[i][0])
            action.append(temp[i][1])
            reward.append(temp[i][2])
            state_next.append(temp[i][3])
            if temp[i][4] == False:
                done.append(np.array(0))
            else:
                done.append(np.array(1))
            decays.append(self.decay)
        return state, action, reward, state_next, done, decays