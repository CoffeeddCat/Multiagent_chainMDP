from env.Env import Env
from model.DQN import DQN
from model.mlp import mlp
import tensorflow as tf
import queue
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread

if __name__ == '__main__':
    #settings:
    ai_number = 4
    n_features = ai_number
    n_actions = 2
    chain_length = 100
    hiddens = [64,128,32]
    EpochLength = 100
    #sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(
            device_count={"CPU": 4},
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
        ))
    C = 0.99
    beta = 0.5
    #f = open('/~/result.txt', 'w')

    # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    left_end_reward = 0.1
    right_end_reward = 10000
    limit_steps = 4000
    limit_episode = 100000

    #try to share some common layers
    common_eval_input = tf.placeholder(tf.float32, shape=[None, n_features], name='common_eval_input')
    common_target_input = tf.placeholder(tf.float32, shape=[None, n_features], name='common_target_input')
    common_eval_output = mlp(inputs=common_eval_input, n_output=64, scope='common_eval_layer', hiddens=hiddens)
    common_target_output = tf.stop_gradient(mlp(inputs=common_eval_input, n_output=64, scope='common_target_layer', hiddens=hiddens))

    #initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis("equal")
    plt.ion()
    #plt.axis([0,5000,0,500])
    x= [0]
    y= [0]

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
    #set environment
    env = Env(chain_length = chain_length,
              agent_number = ai_number,
              left_end_reward = left_end_reward,
              right_end_reward = right_end_reward)

    #set saver
    saver = tf.train.Saver()

    #dataQueue = queue.Queue()
    scoreQueue = queue.Queue()  #not used.

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
            for i in range(ai_number):
                reward[i] = ais[i].return_new_reward(reward = reward[i], state_t=state, state_tpo=state_after, episode=episode, action=action[i])

            #for debug
            total_reward = np.array(reward).sum()
            if steps % 1000 == 0:
                print('action:', action, 'state_after:', state_after, 'reward:', reward, 'totol_reward:', total_reward)

            for i in range(ai_number):
                ais[i].store(state, action[i], reward[i], state_after, episode_end)
                
            state = state_after

            episode_reward += total_reward

            if episode_end:
                need_steps = steps
            #print('step', steps)
            # scoreQueue.put(total_reward)
        print('episode', episode, 'ended, used steps:', steps)

        #update the plot
        # x.append(episode)
        # y.append(steps)
        # ax.plot(x, y, marker='.', c='r')
        # plt.pause(0.001)

        if need_steps < best_steps:
            best_steps = need_steps

        if episode_reward > best_reward:
            best_reward = episode_reward

        if episode % 10 == 0: #every 10 episodes learn
            for i in range(ai_number):
                ais[i].learn()
            print('best rewards:', best_reward, 'best_steps:', best_steps)
            print('now epsilon:', ais[0].epsilon)
        if episode % EpochLength ==0:
            for i in range(ai_number):
                ais[i].update_M()
        if episode % 100 == 0: #every 100 episodes show
            env.reset()
            steps = 0
            episode_end = False
            while steps < limit_steps and not episode_end:
                steps+=1
                action = []
                for i in range(ai_number):
                    action.append(ais[i].check(state))

                state_after, reward, total_reward, episode_end = env.step(action)
                print('action:', action, 'state_after:', state_after, 'reward:', reward)
                state = state_after
            x.append(episode)
            y.append(steps)
            ax.plot(x, y, marker='.', c='r')
            plt.pause(0.001)
            result = 'episode: '+ str(episode) + ' needed steps: ' + str(steps) + '\n'
            #f.write(result)
            print('this is the memory index: ', ais[0].memory.return_index())

        if episode % 1000 ==0: #every 1000 episodes export now
            #haven't done yet.
            continue

    print('exp ended. best reward:', best_reward, 'best_steps:', best_steps)
    f.close()