#settings:
ai_number = 4
n_features = ai_number
n_actions = 2
chain_length = 3
hiddens = [64,64,32]
EpochLength = 100

C = 0.99
beta = 1

left_end_reward = 0.1
right_end_reward = 1000
limit_steps = 4000
limit_episode = 5000

GPU_USED = False
INCENTIVE_USED = True
RESULT_EXPORT = False

SAVE = False
LOAD = False
LOAD_FILE_PATH = ''
RANDOM = False

encoder_output_size = ai_number / 2