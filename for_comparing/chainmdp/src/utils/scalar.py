import tensorflow as tf

class Scalar:
    def __init__(self):
        self.sess = tf.Session()
        self.vars = []
        self.var_names = []

    def add_variable(self, name):
        self.var_names.append(name)

    def set(self):
        for i in range(len(self.var_names)):
            self.vars.append(tf.placeholder(dtype=tf.float32, name=self.var_names[i]))
            tf.summary.scalar(self.var_names[i], self.vars[i])

        self.merged = tf.summary.merge_all()
        self.train_summary = tf.summary.FileWriter('./save/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def read(self, data, step):
        fd = {}
        for i in range(len(self.var_names)):
            fd[self.var_names[i]+':0'] = data[i]
        x = self.sess.run(self.merged, feed_dict=fd)
        self.train_summary.add_summary(x, step)