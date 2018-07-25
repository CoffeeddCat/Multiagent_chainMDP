from env.Env import Env
from model.DQN_withoutE import DQN
from model.mlp import mlp
import tensorflow as tf
import numpy as np
from config import *
import random
from model.auto_encoder import auto_encoder
import copy
from utils.utils import trans_to_one_hot

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
        learning_rate=(1e-2)*5,
        memory_size=2000000,
        batch_size=2000,
        sess=sess,
        output_size = encoder_output_size
    )

    # add agents
    ais = []
    for i in range(ai_number):
        ais.append(DQN(
            n_features=n_features,
            n_actions=n_actions,
            model=mlp,
            hiddens=hiddens,
            scope='agent_' + str(i),
            sess=sess,
            order=i,
            beta=beta,
            C=C
        ))

    # set environment
    env = Env(chain_length=chain_length, agent_number=ai_number, left_end_reward=left_end_reward,
              right_end_reward=right_end_reward)

    episode = 0

    #pretrain for the encoder
    print('pretrain for encoder started.')
    while episode < pretrain_episode:
        episode += 1
        if episode % 1000 == 0:
            print('pretrain episode %d start.' % episode)
        #init for the episode
        state = env.reset()
        steps = 0
        episode_end = False

        #episode start
        while steps < limit_steps and not episode_end:
            steps +=1 
            action = [random.randint(0, 1) for i in range(ai_number)]

            state_after, reward, total_reward, episode_end = env.step(action)

            if steps == limit_steps-1:
                episode_end = True

            encoder.store(trans_to_one_hot(state))

            state = state_after

        if episode % pretrain_update_episode == 0:
            encoder.learn()
            encoder.output_loss()
    print('pretrain for encoder is done.')

    episode = 0
    print('train started.')
    while episode < limit_episode:
        #init
        episode += 1
        print('episode %d start.' % episode)
        state = env.reset()
        steps = 0
        episode_end = False
        episode_reward = 0
        need_steps = limit_steps

        #episode start
        while steps < limit_steps and not episode_end:

            steps +=1 
            action = []

            state_encoded = []
            for i in range(ai_number):
                state_re = copy.deepcopy(state)
                t = state_re[0]
                state_re[0] = state_re[i]
                state_re[i] = t
                state_encoded.append(encoder.output(state_re))

            for i in range(ai_number):
                action.append(ais[i].act(state))

            state_after, reward, total_reward, episode_end = env.step(action)

            state_tpo_encoded = []
            for i in range(ai_number):
                state_re = copy.deepcopy(state_after)
                t = state_re[0]
                state_re[0] = state_re[i]
                state_re[i] = t
                state_tpo_encoded.append(encoder.output(state_re))

            #to gain the new reward
            if INCENTIVE_USED:
                for i in range(ai_number):
                    reward[i] = ais[i].return_new_reward(reward = reward[i], state_t=state_encoded[i], state_tpo=state_tpo_encoded[i], episode=episode, action=action[i])

            #for debug
            total_reward = np.array(reward).sum()
            if steps % 1000 == 0:
                print('action:', action, 'state_after:', state_after, 'reward:', reward, 'totol_reward:', total_reward)

            if steps == limit_steps-1:
                episode_end = True

            for i in range(ai_number):
                ais[i].store(state, action[i], reward[i], state_after, episode_end)
                ais[i].store_encoded(state_encoded[i], action[i], reward[i], state_tpo_encoded[i], episode_end)
                
            state = state_after

            episode_reward += total_reward

            if episode_end:
                need_steps = steps

        order = [i for i in range(ai_number)]

        if episode % 10 == 0: #every 10 episodes learn
            if RANDOM:
                random.shuffle(order)
            for i in order:
                ais[i].learn()
            print('now epsilon:', ais[0].epsilon)

        if episode % 10 == 0:
            if RANDOM:
                random.shuffle(order)
            for i in order:
                ais[i].update_M()

        if episode % 10 == 0: #every 100 episodes show
            state = env.reset()
            steps = 0
            episode_end = False
            r = 0
            while steps < limit_steps and not episode_end:
                steps+=1
                action = []
                state_encoded = encoder.output(state)
                for i in range(ai_number):
                    action.append(ais[i].check(state))

                state_after, reward, total_reward, episode_end = env.step(action)
                print('action:', action, 'state_after:', state_after, 'reward:', reward)
                state = state_after

        if episode % 1000 ==0: #every 1000 episodes export now
            if SAVE:
                saver.save(sess, 'multi-agent chainMDP' ,global_step=episode)
            continue

    print('exp ended. best reward:', best_reward, 'best_steps:', best_steps)
    if RESULT_EXPORT:
        f.close()



