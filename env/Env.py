import copy
import numpy as np

class Env:

    def __init__(self, chain_length, agent_number, left_end_reward, right_end_reward):
        self.chain_length = chain_length
        assert self.chain_length > 0
        self.agent_number = agent_number
        self.state = [0 for i in range(self.agent_number)]
        self.left_end_reward = left_end_reward
        self.right_end_reward = right_end_reward

    def step(self,action):
        '''
        action: an array [] of 1 and 0. 1 means go right, 0 means go left
        
        the function will return: 1.the position(state) after a move of all the agents
                                  2.the reward of all agents
                                  3.total reward
                                  // 4.if the episode is ended
        '''

        state_after = copy.deepcopy(self.state)

        for i in range(self.agent_number):
            move_range_left = 1
            move_range_right = 1
            if self.state[i] == -1:
                move_range_left = 0
            if self.state[i] == self.chain_length -1:
                move_range_right = 0
            if action[i] == 1:
                state_after[i] += move_range_right
            else:
                state_after[i] -= move_range_left
         
        reward = [0 for i in range(self.agent_number)]
        all_right_flag = True
        for i in range(self.agent_number):
            if state_after[i] == -1:
                reward[i] += self.left_end_reward
            if state_after[i] == self.chain_length-1:
                continue
            else:
                all_right_flag = False

        if all_right_flag == True:
            for i in range(self.agent_number):
                reward[i] += self.right_end_reward

        self.state = copy.deepcopy(state_after)

        return state_after, reward, sum(reward),all_right_flag

    def reset(self):
        self.state = [0 for i in range(self.agent_number)]
        reward = [0 for i in range(self.agent_number)]
        return self.state