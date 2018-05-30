import tensorflow as tf

def mlp(inputs, n_output, scope, hiddens, activation=tf.nn.relu):

    with tf.variable_scope(scope):
        out = inputs
        for hidden in hiddens:
            out = tf.contrib.layers.fully_connected(out, num_outputs=hidden, activation_fn = None)
            out = activation(out)

        out = tf.contrib.layers.fully_connected(out, num_outputs=hidden, activation_fn = None)

    return out