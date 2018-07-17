from config import *
from model.mlp import mlp
import tensorflow as tf
import numpy as np
from utils.utils import Memory

class auto_encoder:
    def __init__(
            self,
            learning_rate,
            memory_size
    ):
        self.common_encoder_input = tf.placeholder(tf.float32, shape=[None, n_features], name='common_encoder_input')
        self.common_encoder_output = mlp(inputs=self.common_encoder_input, n_output=n_features, scope='common_encoder_output',
                                    hiddens=[16, 8])
        self.common_decoder_output = mlp(inputs=self.common_encoder_output, n_output=n_features, scope='common_decoder_output')

        self.memory = Memory(memory_size)


    def update(self):

    def store(self):

