import copy

class Env:

    def __init__(self, chain_length, agent_number, left_end_reward, right_end_reward):
        self.chain_length = chain_length
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
        '''

        state_after = copy.deepcopy(self.state)
        for i in range(self.agent_number):
            move_range_left = 1
            move_range_right = 1
            if self.state[i] == -1:
                move_range_left = 0
            if self.state[i] == self.chain_length-1:
                move_range_right = 0
            if action[i] == 1:
                state_after[i] += move_range_right
            else:
                state_after[i] += move_range_left
        
    def reset(self):