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
                                    hiddens=[16, 8])
        self.decoder_output = mlp(inputs=self.encoder_output, n_output=n_features, scope='decoder_output')
        self.encoder_output_ = tf.stop_gradient(self.encoder_output)

        #some const
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size

        #memory
        self.memory = Memory(self.memory_size)

        #for train
        self.loss = tf.reduce_mean(tf.squared_difference(self.encoder_input, self.decoder_output_))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
 
    def train(self):
        data = self.memory.sample(self.batch_size)

        #not sure if the data is legal

        self.sess.run([self.train_0,self.train_1], feed_dict = {
            self.encoder_input: data
        })

    def store(self, state):
        self.memory.store(np.array([state]))

    @property
    def output(self, state):
        return self.sess.run(self.encoder_output_, feed_dict = {
            self.encoder_input: state
            })

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