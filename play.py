from env.Env import Env
from model.DQN import DQN
from model.mlp import mlp
import tensorflow as tf
import numpy as np
from config import *
import random

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

    #try to share some common layers
    common_eval_input = tf.placeholder(tf.float32, shape=[None, n_features], name='common_eval_input')
    common_target_input = tf.placeholder(tf.float32, shape=[None, n_features], name='common_target_input')
    common_eval_output = mlp(inputs=common_eval_input, n_output=64, scope='common_eval_layer', hiddens=hiddens)
    common_target_output = tf.stop_gradient(mlp(inputs=common_eval_input, n_output=64, scope='common_target_layer', hiddens=hiddens))

    #initialize the plot
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.axis("equal")
    # plt.ion()
    # plt.ylim((0,10))
    # x= [0]
    # y= [0]

    #add agents
    ais = []
    for i in range(ai_number):
        ais.append(DQN(
            n_features = n_features,
            n_actions = n_actions,
            model = mlp,
            hiddens = hiddens,
            scope = 'number_' + str(i),
            sess = sess,
            order = i,
            beta = beta,
            C = C,
            common_eval_input = common_eval_input,
            common_target_input = common_target_input,
            common_eval_output = common_eval_output,
            common_target_output = common_target_output
            ))

    writer = tf.summary.FileWriter("logs/", sess.graph)

    #set environment
    env = Env(chain_length = chain_length,
              agent_number = ai_number,
              left_end_reward = left_end_reward,
              right_end_reward = right_end_reward)

    #start explore
    episode = 0
    best_steps = limit_steps
    best_reward = 0

    while episode < limit_episode:
        print('episode', episode, 'start')
        episode += 1
        state = env.reset()
        steps = 0
        episode_reward = 0
        need_steps = limit_steps
        episode_end = False

        while steps < limit_steps and not episode_end:

            steps +=1 
            action = []
            for i in range(ai_number):
                action.append(ais[i].act(state))

            state_after, reward, total_reward, episode_end = env.step(action)

            #to gain the new reward
            if INCENTIVE_USED:
                for i in range(ai_number):
                    reward[i] = ais[i].return_new_reward(reward = reward[i], state_t=state, state_tpo=state_after, episode=episode, action=action[i])

            #for debug
            total_reward = np.array(reward).sum()
            if steps % 1000 == 0:
                print('action:', action, 'state_after:', state_after, 'reward:', reward, 'totol_reward:', total_reward)

            if steps == limit_steps-1:
                episode_end = True

            for i in range(ai_number):
                ais[i].store(state, action[i], reward[i], state_after, episode_end)
                
            state = state_after

            episode_reward += total_reward

            if episode_end:
                need_steps = steps

        print('episode', episode, 'ended, used steps:', steps)

        if need_steps < best_steps:
            best_steps = need_steps

        if episode_reward > best_reward:
            best_reward = episode_reward

        order = [i for i in range(ai_number)]

        if episode % 10 == 0: #every 10 episodes learn
            if RANDOM:
                random.shuffle(order)
            for i in order:
                ais[i].learn()
            print('best rewards:', best_reward, 'best_steps:', best_steps)
            print('now epsilon:', ais[0].epsilon)

        if episode % 10==0:
            if RANDOM:
                random.shuffle(order)
            for i in order:
                ais[i].update_M()
                ais[i].update_encoder()

        if episode % 10 == 0: #every 100 episodes show
            env.reset()
            steps = 0
            episode_end = False
            r = 0
            while steps < limit_steps and not episode_end:
                steps+=1
                action = []
                for i in range(ai_number):
                    action.append(ais[i].check(state))

                state_after, reward, total_reward, episode_end = env.step(action)
                r = r + reward[0]
                print('action:', action, 'state_after:', state_after, 'reward:', reward)
                state = state_after

            #for the plot
            # x.append(episode)
            # y.append(r)
            # ax.plot(x, y, marker='.', c='r')
            # plt.pause(0.001)

            if RESULT_EXPORT:
                result = 'episode: '+ str(episode) + ' needed steps: ' + str(steps) + '\n'
                f.write(result)

            print('this is the memory index: ', ais[0].memory.return_index())

        if episode % 1000 ==0: #every 1000 episodes export now
            if SAVE:
                saver.save(sess, 'multi-agent chainMDP' ,global_step=episode)
            continue

    print('exp ended. best reward:', best_reward, 'best_steps:', best_steps)
    if RESULT_EXPORT:
        f.close()