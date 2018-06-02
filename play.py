from env.Env import Env
from model.DQN import DQN
from model.mlp import mlp

if __name__ == '__main__':
    ais = []
    ai_number = 4
    n_features = ai_number
    n_actions = 2
    hiddens = [64,128,128,32]
    sess = tf.Session()

    for i in range(ai_number):
        ais.append(DQN(
            n_features = n_features,
            n_actions = n_actions,
            model = mlp,
            hiddens = hiddens,
            scope = 'number_' + str(i),
            sess = sess
            ))
    env = Env()
    
    saver = tf.Saver()

    #start explore
    episode = 0
    while True:
        episode += 1
        