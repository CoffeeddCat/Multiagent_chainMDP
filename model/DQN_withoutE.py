import tensorflow as tf
import numpy as np
import random
import copy
from model.mlp import mlp
from config import *

from utils.utils import Memory


class DQN:
    def __init__(
            self,
            n_features,
            n_actions,
            model,
            scope,
            sess,
            order,
            hiddens,
            beta,
            C,
            state_input,
            encoder_output,
            learning_rate=1e-5,
            decay=0.99,
            memory_size=20000000,
            batch_size=100000,
            epsilon_decrement=0.0005,
            epsilon_lower=0.2
    ):
        self.sess = sess
        self.scope = scope
        self.n_features = n_features
        self.batch_size = batch_size
        self.decay = decay
        self.model = model
        self.memory = Memory(memory_size)
        self.order = order
        self.beta = beta
        self.C = C

        self.learn_times = 0

        self.epsilon_lower = epsilon_lower
        self.epsilon_decrement = epsilon_decrement

        #some placeholder for DQN inputs
        self.eval_input = tf.placeholder(tf.float32, shape=[None, self.n_features], name='eval_input')
        self.target_input = tf.placeholder(tf.float32, shape=[None, self.n_features], name='target_input')
        self.actions_selected = tf.placeholder(tf.int32, shape=[None, ], name='actions_selected')
        self.done = tf.placeholder(tf.float32, shape=[None, ], name='done')
        self.decays = tf.placeholder(tf.float32, shape=[None, ], name='decay')
        self.rewards = tf.placeholder(tf.float32, shape=[None, ], name='rewards')

        #some placeholder for M inputs
        self.state_input_tpo = tf.placeholder(tf.float32,shape=[None, self.n_features], name='state_input_t')
        self.state_plus_action_input = tf.placeholder(tf.float32,shape=[None, self.n_features+1], name='action_plus_state_input')

        with tf.variable_scope(self.scope):
            #networks about DQN
            self._epsilon = tf.get_variable(name='epsilon', dtype=tf.float32, initializer=1.0)
            self._epsilon_decrement = tf.constant(epsilon_decrement)
            self.update_epsilon = tf.assign(self._epsilon, self._epsilon - self._epsilon_decrement)
            self.reset_epsilon = tf.assign(self._epsilon, 1)

            self.eval_output = model(inputs=self.encoder_output, n_output=n_actions, scope='eval_net',
                                     hiddens=hiddens)
            self.target_output = tf.stop_gradient(
                model(inputs=self.encoder_output, n_output=n_actions, scope='target_net', hiddens=hiddens))

            #networks about M
            self.predict_output = mlp(inputs=self.state_plus_action_input, n_output=64, scope='predict_output', hiddens=[64,32])
            self.predict_mse = tf.reduce_sum(tf.square(self.state_input_tpo - self.predict_output)) * self.n_features
            self.emax = tf.get_variable(name='emax', dtype=tf.float32, initializer=1.0)
            self.update_emax = tf.assign(self.emax, tf.maximum(self.emax, self.predict_mse))
            self.e_normalize = tf.div(self.predict_mse, self.emax)

            self.M_loss = self.predict_mse
            self.train_M = tf.train.AdamOptimizer(learning_rate).minimize(self.M_loss)

        self.eval_output_selected = tf.reduce_sum(
            self.eval_output * tf.one_hot(self.actions_selected, n_actions), axis=1)
        self.eval_output_target = self.rewards + self.decays * tf.reduce_max(self.target_output, axis=1) * (
            1. - self.done)

        self.loss = tf.reduce_mean(tf.squared_difference(self.eval_output_selected, self.eval_output_target))
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/eval_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/target_net')

        self.update = [tf.assign(x, y) for x, y in zip(self.target_params, self.eval_params)]

        self.sess.run(tf.global_variables_initializer())

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 1)
        else:
            copy_state = copy.deepcopy(state)
            # exchange
            t = copy_state[self.order]
            copy_state[self.order] = copy_state[0]
            copy_state[0] = t

            action = self.sess.run(self.eval_output, feed_dict={
                self.common_eval_input: np.array([copy_state])
            })

            return np.argmax(action, axis=1)[0].tolist()

    def check(self, state):
        copy_state = copy.deepcopy(state)

        # exchange
        t = copy_state[self.order]
        copy_state[self.order] = copy_state[0]
        copy_state[0] = t

        action = self.sess.run(self.eval_output, feed_dict={
            self.common_eval_input: np.array([copy_state])
        })
        return np.argmax(action, axis=1)[0].tolist()

    def learn(self):
        self.learn_times += 1
        state, action, reward, state_next, done, decays = self.process_data()
        self.sess.run(self.train, feed_dict={
            self.common_eval_input: state,
            self.actions_selected: action,
            self.rewards: reward,
            self.common_target_input: state_next,
            self.done: done,
            self.decays: decays
        })

        if self.epsilon > self.epsilon_lower:
            self.sess.run(self.update_epsilon)

        if self.learn_times % 10 == 0:
            print('start update target network')
            self.sess.run(self.update)

    def store(self, state, action, reward, state_after, episode_ended):

        state_copy = copy.deepcopy(state)
        state_after_copy = copy.deepcopy(state_after)
        # exchange
        t = state_copy[self.order]
        state_copy[self.order] = state_copy[0]
        state_copy[0] = t

        t = state_after_copy[self.order]
        state_after_copy[self.order] = state_after_copy[0]
        state_after_copy[0] = t

        self.memory.store(np.array([state_copy, action, reward, state_after_copy, episode_ended]))

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

    @property
    def epsilon(self):
        return self.sess.run(self._epsilon)

    def return_new_reward(self, reward, state_t, state_tpo, episode, action):
        self.sess.run(self.update_emax, feed_dict={
            self.state_input_t: np.array([state_t]),
            self.state_input_tpo: np.array([state_tpo]),
            self.action_plus_state_input: np.array([state_t + [action]])
        })

        temp = self.sess.run(self.e_normalize, feed_dict={
            self.state_input_t: np.array([state_t]),
            self.state_input_tpo: np.array([state_tpo]),
            self.action_plus_state_input: np.array([state_t + [action]]),
        })

        return reward + (self.beta / self.C) * temp

    def update_M(self):
        state, action, reward, state_next, done, decays = self.process_data()
        self.sess.run(self.train_M, feed_dict={
            self.state_input_tpo: state_next,
            self.action_plus_state_input: np.hstack((state, np.array([action]).T))
        })
