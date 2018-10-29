import time

# ====== About the network =========
HIDDENS = [64, 128, 256, 128]
QUANTITY_FEATURES = 60
QUANTITY_ACTIONS = 2
USING_GAN = True

# ====== About communications ======
N_GAMES = 1

# ===== About the scene ===========
MAX_FRAMES_PER_EPISODE = 100
HISTORY_LENGTH = 10

# ====== Save and restore ========
SAVE_EVERY = 500
SAVE_PATH = 'saved_models/' + time.ctime() + '/'

RESTORE = False
RESET_EPSILON = True
ONLY_PLAY = False
RESTORE_PATH = 'saved_models/beat_defensive_ai_2ed/2001.ckpt'
LEARN_START = 0
