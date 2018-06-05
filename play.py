from env.Env import Env
from model.DQN import DQN
from model.mlp import mlp
import tensorflow as tf
import queue

if __name__ == '__main__':
    #settings:
    ais = []
    ai_number = 4
    n_features = ai_number
    n_actions = 2
    hiddens = [64,128,128,32]
    sess = tf.Session()
    left_end_reward = 0.1
    right_end_reward = 100
    limit_steps = 400
    limit_episode = 10000

    #add agents
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
    env = Env(chain_length = 20,
              agent_number = ai_number,
              left_end_reward = left_end_reward,
              right_end_reward = right_end_reward)

    #set saver
    saver = tf.train.Saver()

    #dataQueue = queue.Queue()
    scoreQueue = queue.Queue()

    #start explore
    episode = 0
    while episode < limit_episode:
        episode += 1
        state = env.reset()
        steps = 0

        while steps < limit_steps:

            steps +=1 
            action = []
            for i in range(ai_number):
                action.append(ais[i].act(state))
            state_after, reward, total_reward = env.step(action)

            for i in range(ai_number):
                ais[i].store(state, action[i], reward[i], state_after)
                
            state = state_after
            # scoreQueue.put(total_reward)
        if episode % 100 == 0: #every 100 epsiodes learn
            for i in range(ai_number):
                ais[i].learn()

        if episode % 1000 ==0: #every 1000 episodes export now
            #haven't done yet.
            continue

