import random
import tensorflow as tf

from ..utils.training_utils import *


class DeepQNetwork:
    def __init__(
        self,
        n_features,
        n_actions,
        sess,
        model,
        hiddens,
        parent_ai=None,
        scope='DeepQNetwork',
        learning_rate=1e-3,
        n_replace_target=50,
        decay=1.0,
        memory_size=100000,
        batch_size=2000,
        epsilon_decrement=0.0005,
        epsilon_lower=0.2,
        learn_start=200
    ):

        self.n_actions = n_actions
        self.n_replace_target = n_replace_target
        self.batch_size = batch_size
        self.decay = decay

        self.epsilon_lower = epsilon_lower
        self.learn_start = learn_start

        self.learn_step = 0
        self.sess = sess
        self.eval_input = tf.placeholder(tf.float32, shape=[None, n_features], name='eval_input')
        self.target_input = tf.placeholder(tf.float32, shape=[None, n_features], name='target_input')
        self.actions_selected = tf.placeholder(tf.int32, shape=[None, ], name='actions_selected')
        self.done = tf.placeholder(tf.float32, shape=[None, ], name='done')
        self.rewards = tf.placeholder(tf.float32, shape=[None, ], name='rewards')
        self.decays = tf.placeholder(tf.float32, shape=[None, ], name='decay')

        with tf.variable_scope(scope):
            self._epsilon = tf.get_variable(name='epsilon', dtype=tf.float32, initializer=1.0)
            self._epsilon_decrement = tf.constant(epsilon_decrement)
            self.update_epsilon = tf.assign(self._epsilon, self._epsilon - self._epsilon_decrement)
            self.reset_epsilon = tf.assign(self._epsilon, 1)

            self.eval_output = model(inputs=self.eval_input, n_output=n_actions, scope='eval_net', hiddens=hiddens)
            self.target_output = tf.stop_gradient(
                model(inputs=self.target_input, n_output=n_actions, scope='target_net', hiddens=hiddens))

        self.eval_output_selected = tf.reduce_sum(
            self.eval_output * tf.one_hot(self.actions_selected, n_actions), axis=1)
        self.eval_output_target = self.rewards + \
                                  self.decays * tf.reduce_max(self.target_output, axis=1) * (1. - self.done)

        self.loss = tf.reduce_mean(tf.squared_difference(self.eval_output_selected, self.eval_output_target))
        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/eval_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/target_net')

        self.update = [tf.assign(x, y) for x, y in zip(self.target_params, self.eval_params)]

        if parent_ai is None:
            self.sess.run(tf.global_variables_initializer())
            self.memory = Memory(capacity=memory_size)
            self.sess.run(self.update)
        else:
            self._sync = [tf.assign(x, y) for x, y in zip(self.eval_params, parent_ai.eval_params)] + \
                         [tf.assign(self._epsilon, parent_ai._epsilon)]
            self.parent_ai = parent_ai

    def act(self, s):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        action_values = self.sess.run(self.eval_output, feed_dict={self.eval_input: s[np.newaxis, :]})
        return np.argmax(action_values, axis=1)[0]

    def store(self, exp):
        self.memory.store(exp)

    def learn(self):
        # if self.learn_step % 100 == 1:
        #     print(self.memory.index, self.batch_size * 10)
        #     print(self.learn_step, self.learn_start)

        if self.memory.index < self.batch_size * 10 or self.learn_step < self.learn_start:
            return
        self.learn_step += 1

        s, a, r, s_next, done, decays = self._process_data(self.memory.sample(self.batch_size))

        self.sess.run(self.train, feed_dict={
            self.eval_input: s,
            self.actions_selected: a,
            self.rewards: r,
            self.target_input: s_next,
            self.done: done,
            self.decays: decays
        })

        if self.learn_step % self.n_replace_target == 0:
            self.sess.run(self.update)

        if self.epsilon > self.epsilon_lower:
            self.sess.run(self.update_epsilon)

    def sync(self):
        self.sess.run(self._sync)
        self.learn_step = self.parent_ai.learn_step

    @property
    def epsilon(self):
        return self.sess.run(self._epsilon)

    def _process_data(self, batch_data):
        s, a, r, s_next, done, decays = [], [], [], [], [], []
        for i in range(self.batch_size):
            end_state = batch_data[i][-1][3]
            decay = 1.
            later_reward = 0
            for j in reversed(range(batch_data[0].shape[0])):
                s.append(batch_data[i][j][0])
                a.append(batch_data[i][j][1])
                later_reward = self.decay * later_reward + batch_data[i][j][2]
                r.append(later_reward)
                s_next.append(end_state)
                done.append(batch_data[i][j][4])
                decay = decay * self.decay
                decays.append(decay)

        return s, a, r, s_next, done, decays