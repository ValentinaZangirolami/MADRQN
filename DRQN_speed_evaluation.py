import numpy as np
import os
import airsim
import time
from AirsimEnv.DRQN_2agent_speed import Agent, AirSimWrapper, Qnetwork
import tensorflow as tf
from AirsimEnv.DRQN_2agent_speed import (INPUT_SHAPE, NUM_ACTIONS_1, NUM_ACTIONS_2, DOUBLEDUELING)
import pandas as pd
import random
import rootpath

def conf_dir(env_key, default_value):
    p = os.path.expanduser(os.getenv(env_key, default_value))
    return rootpath.detect(__file__, "^.git$")+p[1:] if p.startswith("./") else p

DATA_HOME = conf_dir('PC_DATA_HOME', "./data/ext/home")
DATA_HOST = conf_dir('PC_DATA_HOST', "./data/ext/host")
DATA_USER = conf_dir('PC_DATA_USER', "./data/ext/user")
DATA_DESK = conf_dir('PC_DATA_DESK', "~/Desktop")


IP = "127.0.0.1"
PORT = 41451

TRAIN_STARTING_POINTS = [(88, -1, 0.2, 1, 0, 0, 0),
                         (127.5, 45, 0.2, 0.7, 0, 0, 0.7),
                         (30, 127.3, 0.2, 1, 0, 0, 0),
                         (-59.5, 126, 0.2, 0, 0, 0, 1),
                         (-127.2, 28, 0.2, 0.7, 0, 0, 0.7),
                         (-129, -48, 0.2, 0.7, 0, 0, -0.7),
                         (-90, -128.5, 0.2, 0, 0, 0, 1),
                         (0, -86, 0.2, 0.7, 0, 0, -0.7),
                         (62, -128.3, 0.2, 1, 0, 0, 0),
                         (127, -73, 0.2, 0.7, 0, 0, -0.7)]

TEST_STARTING_POINTS = [ (0.5,44,0.2,0.7,0,0,0.7),
                        (-75, -0.8, 0.2, 0, 0,0,1),
                        (-128.2, 45, 0.2, 0.7, 0, 0, 0.7),
                        (-0.5,-20, 0.2, 0.7, 0,0, -0.7),
                        (127, -38, 0.2, 0.7, 0, 0, 0.7),
                         (-6, 126.5,0,0,0,0,1),
                         (22, -127.5, 0.2, 1, 0, 0, 0),
                         (126.8,15,0.2,0.7,0,0,-0.7),
                         (-127.2,16,0.2,0.7,0,0,-0.7),
                         (-27,0,0.2,1,0,0,0)]

h_size = 512
EVALUATION_DURING_TRAINING = False

def evaluation_agent(path, num_evaluation, h_size, starting_points):

    df = pd.DataFrame(columns=["point", "reward", "time (s)", "frame"])
    airsim_wrapper = AirSimWrapper(ip=IP, port=PORT, input_shape=INPUT_SHAPE)
    tf.compat.v1.reset_default_graph()
    cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    cellT = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    cellT_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
    mainQN = Qnetwork(h_size, cell, 'main', num_action=NUM_ACTIONS_1, double_dueling=DOUBLEDUELING)
    targetQN = Qnetwork(h_size, cellT, 'target', num_action=NUM_ACTIONS_1, double_dueling=DOUBLEDUELING)
    main_speed_QN = Qnetwork(h_size, cell_2, 'main_2', num_action=NUM_ACTIONS_2, double_dueling=DOUBLEDUELING)
    target_speed_QN = Qnetwork(h_size, cellT_2, 'target_2', num_action=NUM_ACTIONS_2, double_dueling=DOUBLEDUELING)

    init = tf.compat.v1.global_variables_initializer()

    saver = tf.compat.v1.train.Saver(max_to_keep=5)
    replay_memory = ""
    beta= ""
    average = ""
    sess = ""
    agent = Agent(replay_memory, beta, average, beta, average, num_actions_1=NUM_ACTIONS_1, num_actions_2=NUM_ACTIONS_2, input_shape=INPUT_SHAPE)
    with tf.compat.v1.Session() as session:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(session, ckpt.model_checkpoint_path)
        action_list_1_eval = []
        action_list_2_eval = []
        for point in starting_points:
            print("Evaluation from: ", point)
            terminal = True
            state_in = (np.zeros([1, h_size]), np.zeros([1, h_size]))
            state_in_2 = (np.zeros([1, h_size]), np.zeros([1, h_size]))
            for _ in range(num_evaluation):
                while True:
                    if terminal:
                        start_time = time.time()
                        airsim_wrapper.reset(point)
                        episode_reward_sum_1 = 0
                        episode_reward_sum_2 = 0
                        speed_list = []
                        frame_episode = 0
                        terminal = False

                    # Step action
                    action_1, action_2, state1, state2 = agent.get_action(0, mainQN, main_speed_QN, airsim_wrapper.state, state_in, state_in_2, session=session, eval=True)
                    _, reward_1, reward_2, terminal = airsim_wrapper.step(action_1, action_2)
                    speed = airsim_wrapper.env.client.getCarState().speed
                    speed_list.append(speed)
                    action_list_1_eval.append(action_1)
                    action_list_2_eval.append(action_2)
                    frame_episode += 1
                    episode_reward_sum_1 += reward_1
                    episode_reward_sum_2 += reward_2

                    state_in = state1

                    if frame_episode == 2000:
                        terminal = True

                    # On game-over
                    if terminal:
                        df = df.append(
                            {"point": point, 'reward_1': episode_reward_sum_1, 'reward_2': episode_reward_sum_2, 'time (s)': time.time() - start_time,
                             'frame': frame_episode, 'mean_speed': np.mean(speed_list), 'max_speed': np.max(speed_list), 'min_speed': np.min(speed_list), 'std_speed': np.std(speed_list)},
                            ignore_index=True)
                        print({"point": point, 'reward_1': episode_reward_sum_1, 'reward_2': episode_reward_sum_2, 'time (s)': time.time() - start_time,
                               'frame': frame_episode, 'mean_speed': np.mean(speed_list), 'max_speed': np.max(speed_list), 'min_speed': np.min(speed_list), 'std_speed': np.std(speed_list)})
                        break
    return df, action_list_1_eval, action_list_2_eval

if __name__ == "__main__":
    # simulatore 500x300

    if EVALUATION_DURING_TRAINING == False:
        models = ["DRQN_speed"]
        paths = dict()
        paths[models[0]] = "/home/vz21081/data/user/dd/csp-drive-rl.vol/DRL/DRQN_2agent/save-01043784/"


        for model in models:

            print("Evaluation ", model)
            if "D3QN" in model:
                type_network = "D3QN"
            else:
                type_network = "DRQN"

            df_train, action_1_train, action_2_train = evaluation_agent(paths[model], num_evaluation=30, h_size=512, starting_points=TRAIN_STARTING_POINTS)
            df_train.to_csv(DATA_USER + "/DRL/results_DRL/definitivo/" + model + "_train.csv", sep=";")
            np.savez(DATA_USER + "/DRL/results_DRL/action/" + model + "_action_train", action1=action_1_train, action2=action_2_train)


            df_test, action_1_test, action_2_test = evaluation_agent(paths[model], num_evaluation=30, h_size=512, starting_points=TEST_STARTING_POINTS)
            df_test.to_csv(DATA_USER + "/DRL/results_DRL/definitivo/" + model + "_test.csv", sep=";")
            np.savez(DATA_USER + "/DRL/results_DRL/action/" + model + "_action_test", action1=action_1_test, action2=action_2_test)
    else:
        model = "DRQN_bayes"
        type_network = "DRQN"
        paths = {}
        paths[0] = "C:/Users/valen/Desktop/magistrale/tesi/csp-drive-rl-master/DRQN_bayes/save-00563706/main_dqn.h5"
        paths[1] = "C:/Users/valen/Desktop/magistrale/tesi/csp-drive-rl-master/DRQN_bayes/save-00687082/main_dqn.h5"
        paths[2] = "C:/Users/valen/Desktop/magistrale/tesi/csp-drive-rl-master/DRQN_bayes/save-00730097/main_dqn.h5"

        for i in range(3):
            df_test = evaluation_agent(paths[i], num_evaluation=30, h_size=512, starting_points=TEST_STARTING_POINTS)
            df_test.to_csv(
                DATA_USER + "/DRL/results_DRL/" + model + "_" + str(
                    i) + "_test.csv", sep=";")
