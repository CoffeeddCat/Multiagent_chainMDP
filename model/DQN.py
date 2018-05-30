from model.mlp import mlp
import tensorflow as tf
import numpy as np
import random

class DQN:
    def __init__(
        self,
        n_features,
        n_actions,
        model,
        learning_rate=1e-3,
        hiddens,
        decay=0.99,
        memory_size=100000,
        batch_size=2000,
        epsilon_decrement=0.0005,
        epsilon_lower=0.2,
        scope
    ):
        self.sess = tf.Session()
        self.scope = scope
        self.n_features = n_features
        self.batch_size = batch_size
        self.decay = decay
        self.model = mlp

        self.epsilon_lower = epsilon_lower
        self.epsilon_decrement = epsilon_decrement

        self.eval_input = tf.placeholder(tf.float32, shape=[None, self.n_features], name='eval_input')
        self.target_input = tf.placeholder(tf.float32, shape=[None, self.n_features], name='target_input')
        self.rewards = tf.placeholder(tf.float32, shape=[None, ], name='rewards')

        with tf.variable_scope(self.scope):
            self._epsilon = tf.get_variable(name='epsilon', dtype=tf.float32, initializer=1.0)
            self._epsilon_decrement = tf.constant(epsilon_decrement)
            self.update_epsilon = tf.assign(self._epsilon, self._epsilon - self._epsilon_decrement)
            self.reset_epsilon = tf.assign(self._epsilon, 1)
            
            self.eval_output = model(inputs=self.eval_input, n_output=n_actions, scope='eval_net', hiddens=hiddens)
            self.target_output = tf.stop_gradient(
                model(inputs=self.target_input, n_output=n_actions, scope='target_net', hiddens=hiddens))