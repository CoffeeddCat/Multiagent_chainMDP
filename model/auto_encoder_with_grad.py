#this is the encoder with grad but not completed.
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
        self.encoder_input_t = tf.placeholder(tf.float32, shape=[None, n_features], name='encoder_input_t')

        self.encoder_output_t = mlp(inputs=self.encoder_input_t, n_output=output_size, scope='encoder_output_t',
                                    hiddens=[16, 8])
        self.encoder_output_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_output_t')

        self.decoder_output_t = mlp(inputs=self.encoder_output_t, n_output=n_features, scope='decoder_output_t')
        self.decoder_output_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_output_t')

        self.encoder_output_t_ = tf.stop_gradient(self.encoder_output_t)

        #state_t+1  tpo->time plus one
        self.encoder_input_tpo = tf.placeholder(tf.float32, shape=[None, n_features], name='encoder_input_tpo')

        self.encoder_output_tpo = mlp(inputs=self.encoder_input_tpo, n_output=output_size, scope='encoder_output_tpo',
                                    hiddens=[16, 8])
        self.encoder_output_tpo_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder_output_tpo')

        self.decoder_output_tpo = mlp(inputs=self.encoder_output_tpo, n_output=n_features, scope='decoder_output_tpo')
        self.decoder_output_tpo_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder_output_tpo')

        self.encoder_output_tpo_ = tf.stop_gradient(self.encoder_output_tpo)

        #sync
        self.sync_encoder = [tf.assign(x, y) for x, y in zip(self.encoder_output_t_params, self.encoder_output_tpo_params)]
        self.sync_decoder = [tf.assign(x, y) for x, y in zip(self.decoder_output_t_params, self.decoder_output_tpo_params)]

        #some const
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size

        #memory
        self.memory = Memory(self.memory_size)

        #for train
        self.loss_0 = tf.reduce_mean(tf.squared_difference(self.encoder_input_t, self.decoder_output_t))
        self.loss_1 = tf.reduce_mean(tf.squared_difference(self.encoder_input_tpo,self.decoder_output_tpo))

        self.train_0 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_0)
        self.train_1 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_1)
 
    def update(self):
        data = self.memory.sample(self.batch_size)

        #not sure if the data is legal

        self.sess.run([self.train_0,self.train_1] feed_dict = {
            self.encoder_input_t: data,
            self.decoder_output_tpo: data
        })

    def store(self, state):
        self.memory.store(np.array([state]))

    @property
    def output(self):
        return

    @property
    def full(self):
        return self.memory.return_index() == self.memory_size-1

    def sync(self):
        self.sess.run([self.sync_encoder, self.sync_decoder])
        return None

    def learn(self):
        self.sess.run([self.train_0, self.train_1], feed_dict = {

            })
        return None