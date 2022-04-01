import numpy as np
import os
import time
import random
import tensorflow as tf
from scipy import stats

from AirsimEnv.bayesian import Beta, Average
from AirsimEnv.DRQN_2agent_speed import ReplayMemory, Agent, AirSimWrapper, Qnetwork
from AirsimEnv.DRQN_2agent_speed import (BATCH_SIZE, DISCOUNT_FACTOR, FRAMES_BETWEEN_EVAL, TRACE_LENGTH, INPUT_SHAPE,
                           LOAD_REPLAY_MEMORY, MEM_SIZE, NUM_ACTIONS_1, NUM_ACTIONS_2,
                           MIN_REPLAY_MEMORY_SIZE, MAX_EPISODE_LENGTH, ALPHA, BETA,
                           TOTAL_FRAMES, DOUBLEDUELING, STARTING_POINTS)

import rootpath

def conf_dir(env_key, default_value):
    p = os.path.expanduser(os.getenv(env_key, default_value))
    return rootpath.detect(__file__, "^.git$")+p[1:] if p.startswith("./") else p


DATA_HOME = conf_dir('PC_DATA_HOME', "./data/ext/home")
DATA_HOST = conf_dir('PC_DATA_HOST', "./data/ext/host")
DATA_USER = conf_dir('PC_DATA_USER', "./data/ext/user")
DATA_DESK = conf_dir('PC_DATA_DESK', "~/Desktop")


#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.compat.v1.InteractiveSession(config=config)

tf.compat.v1.disable_eager_execution()

IP = "127.0.0.1"
PORT = 41451
TYPE_NETWORK = "DRQN_2agent"
TL = False

LOAD_FROM = '/home/vz21081/data/user/dd/csp-drive-rl.vol/DRL/DRQN_2agent/save-00043312/'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)
tf.compat.v1.random.set_random_seed(123)
tf.compat.v1.set_random_seed(123)


SAVE_PATH = DATA_USER + "/DRL/" + TYPE_NETWORK + "/"
TENSORBOARD_DIR = SAVE_PATH + "tensorboard/"

h_size = 512
tau = 0.001

# Update Target Network: tau is a parameter that allow us to update TN with tau% weights of Main network and (1-tau)% weights of TN
# (Credit to Juliani A. for this and the structure of Recurrent CNN)
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    op_holder_2 = []
    for idx, var in enumerate(tfVars[0:total_vars//4]):
        op_holder.append(tfVars[idx+total_vars//4].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//4].value())))
    for idx_2, var_2 in enumerate(tfVars[2*total_vars//4:3*total_vars//4]):
        op_holder_2.append(tfVars[idx_2+3*total_vars//4].assign((var_2.value()*tau) + ((1-tau)*tfVars[idx_2+3*total_vars//4].value())))
    return op_holder, op_holder_2

def updateTarget(op_holder, op_holder_2, sess):
    for op in op_holder:
        sess.run(op)
    for op_2 in op_holder_2:
        sess.run(op_2)
    total_vars = len(tf.compat.v1.trainable_variables())
    a = tf.compat.v1.trainable_variables()[0].eval(session=sess)
    b = tf.compat.v1.trainable_variables()[total_vars//4].eval(session=sess)
    c = tf.compat.v1.trainable_variables()[2*total_vars//4].eval(session=sess)
    d = tf.compat.v1.trainable_variables()[3*total_vars//4].eval(session=sess)
    if a.all() != b.all() or c.all() != d.all():
        print("Target Set Failed")

if __name__ == "__main__":

    print(TENSORBOARD_DIR)

    airsim_wrapper = AirSimWrapper(ip=IP, port=PORT, input_shape=INPUT_SHAPE)
    tf.compat.v1.reset_default_graph()
    # We define the cells for the primary and target q-networks for first agent (steering angle control)
    cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    cellT = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    # We define the cells for the primary and target q-networks for second agent (speed control)
    cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    cellT_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    mainQN = Qnetwork(h_size, cell, 'main', num_action=NUM_ACTIONS_1, double_dueling=DOUBLEDUELING)
    targetQN = Qnetwork(h_size, cellT, 'target', num_action=NUM_ACTIONS_1, double_dueling=DOUBLEDUELING)
    main_speed_QN = Qnetwork(h_size, cell_2, 'main_2', num_action=NUM_ACTIONS_2, double_dueling=DOUBLEDUELING)
    target_speed_QN = Qnetwork(h_size, cellT_2, 'target_2', num_action=NUM_ACTIONS_2, double_dueling=DOUBLEDUELING)

    beta = Beta(ALPHA, BETA)
    average = Average()
    beta_2 = Beta(ALPHA, BETA)
    average_2 = Average()

    init = tf.compat.v1.global_variables_initializer()

    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    trainables = tf.compat.v1.trainable_variables()

    targetOps, targetOps2 = updateTargetGraph(trainables, tau)

    replay_memory = ReplayMemory(buffer_size=MEM_SIZE, input_shape=INPUT_SHAPE)
    agent = Agent(replay_memory, beta, average, beta_2, average_2, num_actions_1=NUM_ACTIONS_1, num_actions_2=NUM_ACTIONS_2, input_shape=INPUT_SHAPE,
                  batch_size=BATCH_SIZE, trace_length=TRACE_LENGTH)
    with tf.compat.v1.Session() as session:
        writer = tf.compat.v1.summary.FileWriter(TENSORBOARD_DIR, session.graph)


        if LOAD_FROM is None:
            frame_number = 0
            rewards_1 = []
            rewards_2 = []
            loss_list_1 = []
            loss_list_2 = []
            action_list_1 = []
            action_list_2 = []
            terminal_frame = []
            speed_list = []
            eval_list_1 = []
            eval_list_2 = []
            session.run(init)
        else:
            print('Loading from', LOAD_FROM)
            eval_list = np.load(SAVE_PATH + '/evaluation.npz', allow_pickle=True)
            eval_list_1 = list(eval_list['eval1'])
            eval_list_2 = list(eval_list['eval2'])
            action_list = np.load(SAVE_PATH + '/action.npz', allow_pickle=True)
            action_list_1 = list(action_list['action1'])
            action_list_2 = list(action_list['action2'])
            speed_list = list(action_list['speed'])
            terminal_frame = list(action_list['frame_terminal'])
            meta = agent.load(LOAD_FROM, LOAD_REPLAY_MEMORY)
            # Apply information loaded from meta
            frame_number = meta['frame_number']
            rewards_1 = meta['rewards_1']
            rewards_2 = meta['rewards_2']
            loss_list_1 = meta['loss_list_1']
            loss_list_2 = meta['loss_list_2']
            ckpt = tf.train.get_checkpoint_state(LOAD_FROM)
            saver.restore(session, ckpt.model_checkpoint_path)

        initial_start_time = time.time()
        try:

            updateTarget(targetOps, targetOps2, session)
            episode_number = 1
            while frame_number < TOTAL_FRAMES:
                # Training
                state_in = (np.zeros([1, h_size]), np.zeros([1, h_size]))
                state_in_2 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
                epoch_frame = 0
                start_time_progress = time.time()

                while epoch_frame < FRAMES_BETWEEN_EVAL:

                    airsim_wrapper.reset(random.choice(STARTING_POINTS))
                    state_buffer = []
                    action_1_buffer = []
                    action_2_buffer = []
                    next_state_buffer = []
                    reward1_buffer = []
                    reward2_buffer = []
                    terminal_buffer = []


                    episode_reward_sum_1 = 0
                    episode_reward_sum_2 = 0
                    j = 0

                    for j in range(MAX_EPISODE_LENGTH):

                        j+=1

                        frame_time = time.time()
                        # get action
                        frame = airsim_wrapper.state
                        action_1, action_2, state1, state2 = agent.get_action(frame_number, mainQN, main_speed_QN, frame, state_in, state_in_2, session=session)
                        action_list_1.append(action_1)
                        action_list_2.append(action_2)

                        # take step
                        next_frame, reward_1, reward_2, terminal = airsim_wrapper.step(action_1, action_2)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum_1 += reward_1
                        episode_reward_sum_2 += reward_2
                        speed = airsim_wrapper.env.client.getCarState().speed
                        speed_list.append(speed)

                        state_in = state1
                        state_in_2 = state2

                        # update eps-bmc
                        if len(agent.replay_memory.action_1) > MIN_REPLAY_MEMORY_SIZE:
                            q_values_1, q_values_2 = agent.value(mainQN, main_speed_QN, next_frame, state_in, state_in_2, session=session)
                            G_Q_1 = reward_1 + DISCOUNT_FACTOR * np.amax(q_values_1)
                            G_U_1 = reward_1 + DISCOUNT_FACTOR * np.mean(q_values_1)
                            G_Q_2 = reward_2 + DISCOUNT_FACTOR * np.amax(q_values_2)
                            G_U_2 = reward_2 + DISCOUNT_FACTOR * np.mean(q_values_2)
                            agent.update_posterior(data_1=(G_Q_1, G_U_1), data_2=(G_Q_2, G_U_2))


                        # add experience
                        if frame.shape != INPUT_SHAPE or next_frame.shape != INPUT_SHAPE:
                            print("Dimension of frame is wrong!")
                        else:
                            state_buffer.append(np.array(np.reshape(frame, (66, 200, 3)), dtype=np.uint8))
                            next_state_buffer.append(np.array(np.reshape(next_frame, (66, 200, 3)), dtype=np.uint8))
                            action_1_buffer.append(action_1)
                            reward1_buffer.append(reward_1)
                            action_2_buffer.append(action_2)
                            reward2_buffer.append(reward_2)
                            terminal_buffer.append(terminal)

                        # update two agents

                        if frame_number % 4 == 0 and len(agent.replay_memory.action_1) > MIN_REPLAY_MEMORY_SIZE:

                            state_train = (np.zeros([BATCH_SIZE, h_size]), np.zeros([BATCH_SIZE, h_size]))
                            updateTarget(targetOps, targetOps2, session)
                            loss1, _ = agent.learn_1(main_drqn_steer=mainQN, target_drqn_steer=targetQN,batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                  frame_number=frame_number, trace_length=TRACE_LENGTH,
                                                  state_train=state_train, session=session)
                            loss2, _ = agent.learn_2(main_drqn_speed=main_speed_QN, target_drqn_speed=target_speed_QN,batch_size=BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                     frame_number=frame_number, trace_length=TRACE_LENGTH,
                                                     state_train=state_train, session=session)
                            loss_list_1.append(loss1)
                            loss_list_2.append(loss2)

                        elif frame_number % 4 == 0:
                            time.sleep(0.10)

                        # Break the loop when the game is over
                        if terminal:
                            terminal = False
                            break

                        #print("Time of frame evaluation:", time.time() - frame_time)

                    rewards_1.append(episode_reward_sum_1)
                    rewards_2.append(episode_reward_sum_2)
                    terminal_frame.append(frame_number)
                    episode_number += 1

                    #add episode to replay memory
                    if j >= TRACE_LENGTH:
                        agent.add_experience(np.array(state_buffer), np.array(action_1_buffer), np.array(action_2_buffer), np.array(reward1_buffer), np.array(reward2_buffer),np.array(next_state_buffer), np.array(terminal_buffer))

                    # Output the progress every 100 games
                    if len(rewards_1) % 100 == 0:

                        hours = divmod(time.time() - initial_start_time, 3600)
                        minutes = divmod(hours[1], 60)
                        minutes_100 = divmod(time.time() - start_time_progress, 60)
                        print(f'Game number: {str(len(rewards_1)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  '
                              f'Average reward_1: {np.mean(rewards_1[-100:]):0.1f} Average reward_2: {np.mean(rewards_2[-100:]):0.1f}  Time taken: {(minutes_100[0]):.1f}  '
                              f'Total time taken: {(int(hours[0]))}:{(int(minutes[0]))}:{(minutes[1]):0.1f} '
                              f'Min: {min(rewards_1[-100:]):0.1f}  Max: {max(rewards_1[-100:]):0.1f} ')
                        start_time_progress = time.time()

                    # Save model
                    if len(rewards_1) % 500 == 0 and SAVE_PATH is not None:

                        agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number,
                                       rewards_1=rewards_1, rewards_2=rewards_2, loss_list_1=loss_list_1, loss_list_2=loss_list_2)
                        saver.save(session, f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}' + '/model.cptk')
                        np.savez(SAVE_PATH + '/action', action1=action_list_1, action2=action_list_2, speed=speed_list, frame_terminal=terminal_frame)

                # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                if frame_number > 0:
                    eval_rewards_1 = []
                    eval_rewards_2 = []
                    evaluate_frame_number = 0

                    terminal = True
                    for point in STARTING_POINTS:

                        state_in = (np.zeros([1, h_size]), np.zeros([1, h_size]))
                        state_in_2 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
                        while True:
                            if terminal:
                                airsim_wrapper.reset(point)
                                episode_reward_sum_1 = 0
                                episode_reward_sum_2 = 0
                                frame_episode = 0
                                terminal = False

                        # Step action
                            action_1, action_2, state1, state2 = agent.get_action(frame_number, mainQN, main_speed_QN, airsim_wrapper.state, state_in, state_in_2, session=session, eval=True)
                            _, reward_1, reward_2, terminal = airsim_wrapper.step(action_1, action_2)
                            evaluate_frame_number += 1
                            frame_episode += 1
                            episode_reward_sum_1 += reward_1
                            episode_reward_sum_2 += reward_2
                            state_in = state1
                            state_in_2 = state2

                        # On game-over
                            if terminal:
                                print("Reward 1 per episode: ", episode_reward_sum_1)
                                print("Reward 2 per episode: ", episode_reward_sum_2)
                                eval_rewards_1.append(episode_reward_sum_1)
                                eval_rewards_2.append(episode_reward_sum_2)
                                break

                    if len(eval_rewards_1) > 0:
                        final_score_1 = np.mean(eval_rewards_1)
                        final_score_2 = np.mean(eval_rewards_2)
                    else:
                    # In case the game is longer than the number of frames allowed
                        final_score_1 = episode_reward_sum_1
                        final_score_2 = episode_reward_sum_2
                    # Print score and write to tensorboard

                    print('Evaluation score 1:', final_score_1)
                    print('Evaluation score 2:', final_score_2)
                    eval_list_1.append(final_score_1)
                    eval_list_2.append(final_score_2)
                    np.savez(SAVE_PATH + '/evaluation', eval1=eval_list_1, eval2=eval_list_2)


            agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number,
                           rewards_1=rewards_1, rewards_2=rewards_2, loss_list_1=loss_list_1, loss_list_2=loss_list_2)
            saver.save(session, f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}' + '/model.cptk')
            np.savez(SAVE_PATH + '/action', action1=action_list_1, action2=action_list_2, speed=speed_list, frame_terminal=terminal_frame)


        except KeyboardInterrupt:
            print('\nTraining exited early.')
            writer.close()

            if SAVE_PATH is None:
                try:
                    SAVE_PATH = input(
                    'Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
                except KeyboardInterrupt:
                    print('\nExiting...')

            if SAVE_PATH is not None:
                print('Saving...')
                agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number,
                                       rewards_1=rewards_1, rewards_2=rewards_2, loss_list_1=loss_list_1, loss_list_2=loss_list_2)

                saver.save(session, f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}' + '/model.cptk')
                np.savez(SAVE_PATH + '/action', action1=action_list_1, action2=action_list_2, speed=speed_list, frame_terminal=terminal_frame)
                print('Saved.')
