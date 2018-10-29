import tensorflow.contrib.layers as layers
import tensorflow as tf


def mlp(inputs, n_output, scope, hiddens, activation=tf.nn.relu):

    with tf.variable_scope(scope):
        out = inputs
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            out = activation(out)

        out = layers.fully_connected(out, num_outputs=n_output, activation_fn=None)

    return out


def cnn_to_mlp(convs, hiddens, inputs, num_actions, scope, dueling=True):

    with tf.variable_scope(scope):
        out = inputs
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def generator(z, num_outputs, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        hidden1 = tf.nn.leaky_relu(tf.layers.dense(inputs=z, units=50), alpha=0.01)
        hidden2 = tf.nn.leaky_relu(tf.layers.dense(inputs=hidden1, units=50), alpha=0.01)
        output = tf.layers.dense(hidden2, units=num_outputs, activation=tf.nn.tanh)
        return output


def discriminator(X, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        hidden1 = tf.nn.leaky_relu(tf.layers.dense(inputs=X, units=50), alpha=0.01)
        hidden2 = tf.nn.leaky_relu(tf.layers.dense(inputs=hidden1, units=50), alpha=0.01)
        logits = tf.layers.dense(hidden2, units=1)
        output = tf.sigmoid(logits)

        return output, logits
