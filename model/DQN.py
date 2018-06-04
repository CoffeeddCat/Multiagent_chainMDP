import tensorflow as tf
import numpy as np
import random
import queue
import copy

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
        scope,
        sess,
        order
    ):
        self.sess = sess
        self.scope = scope
        self.n_features = n_features
        self.batch_size = batch_size
        self.decay = decay
        self.model = mlp
        self.memory = queue.Queue()
        self.order = order

        self.epsilon_lower = epsilon_lower
        self.epsilon_decrement = epsilon_decrement

        self.eval_input = tf.placeholder(tf.float32, shape=[None, self.n_features], name='eval_input')
        self.target_input = tf.placeholder(tf.float32, shape=[None, self.n_features], name='target_input')
        self.done = tf.placeholder(tf.float32, shape=[None, ], name='done')
        self.decays = tf.placeholder(tf.float32, shape=[None, ], name='decay')
        self.rewards = tf.placeholder(tf.float32, shape=[None, ], name='rewards')

        with tf.variable_scope(self.scope):
            self._epsilon = tf.get_variable(name='epsilon', dtype=tf.float32, initializer=1.0)
            self._epsilon_decrement = tf.constant(epsilon_decrement)
            self.update_epsilon = tf.assign(self._epsilon, self._epsilon - self._epsilon_decrement)
            self.reset_epsilon = tf.assign(self._epsilon, 1)
            
            self.eval_output = model(inputs=self.eval_input, n_output=n_actions, scope='eval_net', hiddens=hiddens)
            self.target_output = tf.stop_gradient(
                model(inputs=self.target_input, n_output=n_actions, scope='target_net', hiddens=hiddens))

        self.eval_output_selected = tf.reduce_sum(
            self.eval_output * tf.one_hot(self.actions_selected, n_actions), axis=1)
        self.eval_output_target = self.rewards + self.decays * tf.reduce_max(self.target_output, axis=1) * (1. - self.done)

        self.loss = tf.reduce_mean(tf.squared_difference(self.eval_output_selected, self.eval_output_target))
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/eval_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/target_net')

        self.update = [tf.assign(x, y) for x, y in zip(self.target_params, self.eval_params)]

        self.sess.run(tf.global_variables_initializer())

    def act(self,state):
        copy_state = copy.deepcopy(state)

        #exchange
        t = copy_state[self.order]
        copy_state[self.order] = copy_state[0]
        copy_state[0] = t

        action = self.sess.run(self.eval_output, feed_dict={
            self.eval_input: np.array(copy_state)
            })

        return action.tolist()

    def learn(self):

    def store(self, data):
        store_data = copy.deepcopy(data)

        #exchange
        t = store_data[self.order]
        store_data[self.order] = store_data[0]
        store_data[0] = t

        self.memory.put(store_data)

    def process_data(self):
        state, action, reward, state_next, done, decays = [], [], [], [], [], []
        temp_data = np.random.choice(self.memory, batch_size)
        for i in range(self.batch_size):
            
