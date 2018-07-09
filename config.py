#settings:
ai_number = 1
n_features = ai_number
n_actions = 2
chain_length = 10
hiddens = [64,64,32]
EpochLength = 100

C = 0.99
beta = 0.5

left_end_reward = 0
right_end_reward = 100
limit_steps = 4000
limit_episode = 1000

GPU_USED = False
INCENTIVE_USED = False
RESULT_EXPORT = False