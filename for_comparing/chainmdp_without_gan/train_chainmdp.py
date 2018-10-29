from src.env.chain_mdp import ChainMDP
from src.utils.training_utils import *
from src.models.models import *
from src.models.learn import DeepQNetwork
from src.models.gan import GAN
from src.constants.config import *
from src.utils.scalar import *

import queue
import tensorflow as tf
import threading as td

def play(self_ai, parent_ai, gan, data_queue, score_queue):

    env = ChainMDP(QUANTITY_FEATURES)
    history = History(HISTORY_LENGTH)
    episode = 0
    score_sum = 0
    scalar = Scalar()
    scalar.add_variable("Average_score")
    scalar.set()

    while True:
        # ========================================== start a new episode ==========================================
        frame_counter = 0  # record the num of frames which are stored, not include the skipped ones
        synchronize_version(self_ai, parent_ai)
        episode += 1
        s0 = env.reset()

        score = 0  # accumulated reward along the whole episode added from two players
        max_state = 0

        while True:
            # if episode % 100 == 1:
            #     env.render()
            action = self_ai.act(s0)
            reward, s1 = env.step(action)
            max_state = max(env.loc, max_state)

            if episode > 100 and USING_GAN:
                prob = gan.single_state_prob(s1)
                #bonus_reward = (1-2*prob)**2 if prob < 0.5 else 0
                # if bonus_reward > 0.3:
                #     print(env.loc, bonus_reward)
                history.put((s0, action, reward, s1, 0))
            else:
                history.put((s0, action, reward, s1, 0))
            if history.full():
                data_queue.put(history.get())
            s0 = s1
            frame_counter += 1
            score += reward

            # ======================================= when the game ends ==========================================
            if frame_counter >= MAX_FRAMES_PER_EPISODE:
                score_queue.put(score)
                score_sum += score
                if episode % 10 == 1:
                    print()
                    print("Max state:", max_state)
                    print("Score: {:2f}".format(score))
                    print("Epsilon : %f" % self_ai.epsilon)
                    print("Average score: %f" % (score_sum / 10.0))
                    print()
                    scalar.read([score_sum / 10.0], episode)
                    score_sum = 0
                break


if __name__ == '__main__':

    Sess = tf.Session()

    global_ai = DeepQNetwork(
        n_features=QUANTITY_FEATURES,
        n_actions=2,
        scope='global_ai',
        model=mlp,
        parent_ai=None,
        sess=Sess,
        learning_rate=5e-3,
        n_replace_target=50,
        hiddens=HIDDENS,
        decay=0.99,
        memory_size=10000,
        batch_size=300,
        epsilon_decrement=5e-4,
        epsilon_lower=0.02,
        learn_start=LEARN_START,
    )

    if USING_GAN:
        gan = GAN(
            sess=Sess,
            n_features=QUANTITY_FEATURES,
            memory_size=100000,
            batch_size=300,
            generator=generator,
            discriminator=discriminator,
            learning_rate=0.001
        )
    else:
        gan = None

    dataQ = queue.Queue()

    score_plotter = ScorePlotter()
    scoreQ = queue.Queue()

    ais = []
    for i in range(N_GAMES):
        ais.append(
            DeepQNetwork(
                n_features=QUANTITY_FEATURES,
                n_actions=QUANTITY_ACTIONS,
                scope='local_ai_' + str(i),
                model=mlp,
                parent_ai=global_ai,
                sess=Sess,
                hiddens=HIDDENS
            )
        )

    Saver = tf.train.Saver()

    if RESTORE:
        Saver.restore(Sess, RESTORE_PATH)
        if RESET_EPSILON:
            Sess.run(global_ai.reset_epsilon)
        print('restored successfully from ' + RESTORE_PATH)
    else:
        Sess.run(tf.global_variables_initializer())

    for i in range(N_GAMES):
        new_thread = td.Thread(target=play, args=(ais[i], global_ai, gan, dataQ, scoreQ))
        new_thread.start()

    while True:
        plot_score(score_plotter, scoreQ)
        fetch_data(global_ai, gan, dataQ)
        global_ai.learn()
        if USING_GAN:
            D_loss = gan.train_D()
            G_loss = gan.train_G()
            if global_ai.learn_step % 100 == 1:
                print("D loss:", gan.train_D())
                print("G loss:", gan.train_G())
        if not ONLY_PLAY and global_ai.learn_step % SAVE_EVERY == 1:
            save_path = Saver.save(Sess, SAVE_PATH + str(global_ai.learn_step) + '.ckpt')
            print('saved in' + save_path)
