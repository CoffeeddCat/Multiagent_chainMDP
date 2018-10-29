import tensorflow as tf

from ..utils.training_utils import *


class GAN:

    def __init__(self, sess, n_features, batch_size, memory_size, generator, discriminator, learning_rate=0.001):
        self.sess = sess
        self.n_features = n_features

        self.real_states = tf.placeholder(tf.float32, shape=[None, n_features])
        self.z = tf.placeholder(tf.float32, shape=[None, n_features])
        self.G = generator(self.z, num_outputs=n_features)
        self.D_output_real, self.D_logits_real = discriminator(self.real_states)
        self.D_output_fake, self.D_logits_fake = discriminator(self.G, reuse=True)

        def loss_func(logits_in, labels_in):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))

        self.D_real_loss = loss_func(self.D_logits_real, tf.ones_like(self.D_logits_real) * 0.9)
        self.D_fake_loss = loss_func(self.D_logits_fake, tf.zeros_like(self.D_logits_real))
        self.D_loss = self.D_real_loss + self.D_fake_loss

        self.G_loss = loss_func(self.D_logits_fake, tf.ones_like(self.D_logits_fake))

        self.tvars = tf.trainable_variables()
        self.d_vars = [var for var in self.tvars if 'dis' in var.name]
        self.g_vars = [var for var in self.tvars if 'gen' in var.name]

        self.D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(self.D_loss, var_list=self.d_vars)
        self.G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(self.G_loss, var_list=self.g_vars)

        self.memory = Memory(capacity=memory_size)
        self.batch_size = batch_size

    def train_D(self):
        if self.memory.index < self.batch_size:
            return
        real_states = [list(state_array) for state_array in self.memory.sample(self.batch_size)]
        self.sess.run(self.D_trainer, feed_dict={
            self.real_states: real_states, self.z: self.batch_noise()})
        return self.sess.run(self.D_loss, feed_dict={
            self.real_states: real_states, self.z: self.batch_noise()})

    def train_G(self):
        if self.memory.index < self.batch_size:
            return
        self.sess.run(self.G_trainer, feed_dict={self.z: self.batch_noise()})
        return self.sess.run(self.G_loss, feed_dict={self.z: self.batch_noise()})

    def batch_noise(self):
        return np.random.uniform(-1, 1, (self.batch_size, self.n_features))

    def single_state_prob(self, state):
        return self.sess.run(self.D_output_real, feed_dict={self.real_states: state.reshape(-1, self.n_features)})[0][0]

    def store(self, state):
        self.memory.store(state)

