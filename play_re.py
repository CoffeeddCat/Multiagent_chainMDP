from env.Env import Env
from model.DQN_withoutE import DQN
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

    # share the encoder
    encoder = auto_encoder(
        learning_rate=1e-5,
        memory_size=2000000,
        batch_size=2000,
        sess=sess
    )

    # add agents
    ais = []
    for i in range(ai_number):
        ais.append(DQN(
            n_features=n_features,
            n_actions=n_actions,
            model=mlp,
            hiddens=hiddens,
            scope='number_' + str(i),
            sess=sess,
            order=i,
            beta=beta,
            C=C
        ))

    # set environment
    env = Env(chain_length=chain_length, agent_number=ai_number, left_end_reward=left_end_reward,
              right_end_reward=right_end_reward)
