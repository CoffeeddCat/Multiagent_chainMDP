from env.Env import Env
from model.DQN import DQN
from model.mlp import mlp
import tensorflow as tf
import queue

if __name__ == '__main__':
    #settings:
    ai_number = 10
    n_features = ai_number
    n_actions = 2
    chain_length = 20
    hiddens = [64,128,128,32]
    sess = tf.Session()
    left_end_reward = 0.1
    right_end_reward = 100
    limit_steps = 40000
    limit_episode = 100000

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
            order = i
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

        while steps < limit_steps or not episode_end:

            steps +=1 
            action = []
            for i in range(ai_number):
                action.append(ais[i].act(state))

            state_after, reward, total_reward, episode_end = env.step(action)

            #for debug
            if steps % 1000 == 0:
                print('action:', action, 'state_after:', state_after, 'reward:', reward, 'totol_reward:', total_reward)

            for i in range(ai_number):
                ais[i].store(state, action[i], reward[i], state_after)
                
            state = state_after

            episode_reward += total_reward

            if episode_end:
                need_steps = steps
            #print('step', steps)
            # scoreQueue.put(total_reward)

        if need_steps < best_steps:
            best_steps = need_steps

        if episode_reward > best_reward:
            best_reward = episode_reward

        if episode % 100 == 0: #every 100 epsiodes learn
            for i in range(ai_number):
                ais[i].learn()
            print('best rewards:', best_reward, 'best_steps:', best_steps)

        if episode % 1000 ==0: #every 1000 episodes export now
            #haven't done yet.
            continue

    print('exp ended. best reward:', best_reward, 'best_steps:', best_steps)