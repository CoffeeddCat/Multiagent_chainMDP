from env.Env import Env
from model.DQN import DQN
from model.mlp import mlp
import tensorflow as tf
import numpy as np
from config import *
import random
from model.auto_encoder import auto_encoder

if __name__ == '__main__':

    if GPU_USED:
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    else:
        sess = tf.Session(config=tf.ConfigProto(
                device_count={"CPU": 4},
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1,
            ))

    # set saver
    if SAVE:
        saver = tf.train.Saver()
    if LOAD:
        model_file = tf.train.latest_checkpoint(LOAD_FILE_PATH)
        saver.restore(sess, model_file)

    if RESULT_EXPORT:
        f = open('/~/result.txt', 'w')

    #share the encoder
    common_encoder_input = tf.placeholder(tf.float32, shape=[None, n_features], name='common_encoder_input')
    common_encoder_output = mlp(inputs=common_encoder_input, n_output=n_features, scope='common_encoder_output', hiddens=[16,8])
    common_decoder_input = mlp(inputs=common_encoder_output)