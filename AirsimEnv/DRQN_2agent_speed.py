import numpy as np
import random
import os
import time
import tensorflow as tf
import json
import math
import cv2
from scipy.stats import t as tdist
from AirsimEnv.AirsimEnv_speed import AirsimEnv
from AirsimEnv.bayesian import Beta, Average
import tf_slim as slim

TOTAL_FRAMES = 1000000           # Total number of frames to train for
TOTAL_EPISODE = 24000
MAX_EPISODE_LENGTH = 2000        # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes (when process is based on episode, this par. is equal to 10000 (Episode to infinity))
FRAMES_BETWEEN_EVAL = 50000      # Number of frames between evaluations
UPDATE_FREQ = 10000               # Number of actions chosen between updating the target network

TRACE_LENGTH = 10
DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
MIN_REPLAY_MEMORY_SIZE = 999     # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = 1000              # The maximum size of the replay buffer

INPUT_SHAPE = (66, 200, 3)      # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 10   # Number of samples the agent learns from at once
NUM_ACTIONS_1 = 9        #number of actions of steering angle
NUM_ACTIONS_2 = 5        #number of actions of throttle/brake

# Parameters of EPS-BMC
ALPHA = 25.0
BETA = 25.01

DOUBLEDUELING = True

STARTING_POINTS = [(88, -1, 0.2, 1, 0, 0, 0),
                        (127.5, 45, 0.2, 0.7, 0, 0, 0.7),
                        (30, 127.3, 0.2, 1, 0, 0, 0),
                        (-59.5, 126, 0.2, 0, 0, 0, 1),
                        (-127.2, 28, 0.2, 0.7, 0, 0, 0.7),
                        (-129, -48, 0.2, 0.7, 0, 0, -0.7),
                        (-90, -128.5, 0.2, 0, 0, 0, 1),
                        (0, -86, 0.2, 0.7, 0, 0, -0.7),
                        (62, -128.3, 0.2, 1, 0, 0, 0),
                        (127, -73, 0.2, 0.7, 0, 0, -0.7)]

LOAD_REPLAY_MEMORY = True
WRITE_TENSORBOARD = True
WEIGHT_PATH = 'mnith_weights.h5'  # weights of the pre-trained network

class AirSimWrapper:

    def __init__(self, input_shape, ip, port):
        self.env = AirsimEnv(ip, port)
        self.input_shape = input_shape
        self.state = np.empty(input_shape)

    def frameProcessor(self, frame):
        # assert frame.dim == 3
        frame = frame[40:136, 0:255, 0:3]
        frame = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)
        return frame

    def reset(self, starting_point):

        observation = self.env.reset(starting_point)
        time.sleep(0.2)
        self.env.step(0, 4)
        speed = self.env.client.getCarState().speed
        while speed < 3.0:
            speed = self.env.client.getCarState().speed

        frame = self.frameProcessor(observation)
        self.state = frame

    def step(self, action_1, action_2):

        new_frame, reward_1, reward_2, done, info = self.env.step(action_1, action_2)
        processed_frame = self.frameProcessor(new_frame)

        self.state = processed_frame

        return processed_frame, reward_1, reward_2, done

class ReplayMemory:
    def __init__(self, buffer_size, input_shape):

        self.input_shape = input_shape
        self.state = []
        self.action_1 = []
        self.action_2 = []
        self.reward_1 = []
        self.reward_2 = []
        self.next_state = []
        self.terminal = []
        self.buffer_size = buffer_size

    #add state, action_1, action_2, reward_1, reward_2, next_state and terminal into buffer
    def add_experience(self, frames, actions_1, actions_2, rewards_1, rewards_2, next_frames, terminal):
        if len(self.action_1) + 1 >= self.buffer_size:
            self.state[0: (1 + len(self.state)) - self.buffer_size] = []
            self.action_1[0: (1 + len(self.action_1)) - self.buffer_size] = []
            self.reward_1[0: (1 + len(self.reward_1)) - self.buffer_size] = []
            self.action_2[0: (1 + len(self.action_2)) - self.buffer_size] = []
            self.reward_2[0: (1 + len(self.reward_2)) - self.buffer_size] = []
            self.next_state[0: (1 + len(self.next_state)) - self.buffer_size] = []
            self.terminal[0: (1 + len(self.terminal)) - self.buffer_size] = []
        self.state.append(frames)
        self.action_1.append(actions_1)
        self.reward_1.append(rewards_1)
        self.action_2.append(actions_2)
        self.reward_2.append(rewards_2)
        self.next_state.append(next_frames)
        self.terminal.append(terminal)

    #Traces of experience: Take random episode, take a random point into episode and construct a trace with length equal to trace_length
    def sample_1(self, batch_size=BATCH_SIZE, trace_length=TRACE_LENGTH):
        # Sample of indexes for each component of buffer for first agent (steering angle)
        sample_episodes = np.random.randint(0, len(self.action_1), size=batch_size)
        sampled_action = []
        sampled_reward = []
        sampled_state = []
        sampled_nextstate = []
        sampled_terminal = []
        for index in sample_episodes:
            episode_state, episode_action, episode_reward = self.state[index], self.action_1[index], self.reward_1[index]
            episode_nextstate, episode_terminal = self.next_state[index], self.terminal[index]
            # Random point in episode
            point = np.random.randint(0, len(episode_action) + 1 - trace_length)
            sampled_action.append(episode_action[point: point + trace_length])
            sampled_reward.append(episode_reward[point: point + trace_length])
            sampled_state.append(episode_state[point: point + trace_length])
            sampled_nextstate.append(episode_nextstate[point: point + trace_length])
            sampled_terminal.append(episode_terminal[point: point + trace_length])
        sampled_state = np.reshape(np.array(sampled_state), [batch_size * trace_length,  self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        sampled_reward = np.reshape(np.array(sampled_reward), [batch_size*trace_length])
        sampled_nextstate = np.reshape(np.array(sampled_nextstate), [batch_size * trace_length, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        sampled_action = np.reshape(np.array(sampled_action), [batch_size * trace_length])
        sampled_terminal = np.reshape(np.array(sampled_terminal), [batch_size * trace_length])
        return sampled_state, sampled_reward, sampled_action, sampled_nextstate, sampled_terminal

    def sample_2(self, batch_size=BATCH_SIZE, trace_length=TRACE_LENGTH):
        # Sample of indexes for each component of buffer for second agent (speed)
        sample_episodes = np.random.randint(0, len(self.action_2), size=batch_size)
        sampled_action = []
        sampled_reward = []
        sampled_state = []
        sampled_nextstate = []
        sampled_terminal = []
        for index in sample_episodes:
            episode_state, episode_action, episode_reward = self.state[index], self.action_2[index], self.reward_2[index]
            episode_nextstate, episode_terminal = self.next_state[index], self.terminal[index]
            # Random point in episode
            point = np.random.randint(0, len(episode_action) + 1 - trace_length)
            sampled_action.append(episode_action[point: point + trace_length])
            sampled_reward.append(episode_reward[point: point + trace_length])
            sampled_state.append(episode_state[point: point + trace_length])
            sampled_nextstate.append(episode_nextstate[point: point + trace_length])
            sampled_terminal.append(episode_terminal[point: point + trace_length])
        sampled_state = np.reshape(np.array(sampled_state), [batch_size * trace_length,  self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        sampled_reward = np.reshape(np.array(sampled_reward), [batch_size*trace_length])
        sampled_nextstate = np.reshape(np.array(sampled_nextstate), [batch_size * trace_length, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        sampled_action = np.reshape(np.array(sampled_action), [batch_size * trace_length])
        sampled_terminal = np.reshape(np.array(sampled_terminal), [batch_size * trace_length])
        return sampled_state, sampled_reward, sampled_action, sampled_nextstate, sampled_terminal


    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        np.save(folder_name + '/state.npy', self.state)
        np.save(folder_name + '/action_1.npy', self.action_1)
        np.save(folder_name + '/reward_1.npy', self.reward_1)
        np.save(folder_name + '/action_2.npy', self.action_2)
        np.save(folder_name + '/reward_2.npy', self.reward_2)
        np.save(folder_name + '/nextstate.npy', self.next_state)
        np.save(folder_name + '/terminal.npy', self.terminal)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.state = list(np.load(folder_name + '/state.npy', allow_pickle=True))
        self.action_1 = list(np.load(folder_name + '/action_1.npy', allow_pickle=True))
        self.reward_1 = list(np.load(folder_name + '/reward_1.npy', allow_pickle=True))
        self.action_2 = list(np.load(folder_name + '/action_2.npy', allow_pickle=True))
        self.reward_2 = list(np.load(folder_name + '/reward_2.npy', allow_pickle=True))
        self.next_state = list(np.load(folder_name + '/nextstate.npy', allow_pickle=True))
        self.terminal = list(np.load(folder_name + '/terminal.npy', allow_pickle=True))


class Qnetwork:
    def __init__(self, h_size, rnn_cell, myScope, num_action, double_dueling):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.compat.v1.placeholder(shape=[None, 66, 200, 3], dtype=tf.float32)
        self.conv1 = slim.convolution2d(inputs=self.scalarInput, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',biases_initializer=None, scope=myScope +'_conv1')
        self.conv2 = slim.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID', biases_initializer=None, scope=myScope +'_conv2')
        self.conv3 = slim.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID', biases_initializer=None, scope=myScope +'_conv3')
        self.trainLength = tf.compat.v1.placeholder(dtype=tf.int32)
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.batch_size = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])
        self.convFlat = tf.compat.v1.reshape(slim.flatten(self.conv3), [self.batch_size, self.trainLength, slim.flatten(self.conv3).shape[1]])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.compat.v1.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
        self.rnn = tf.compat.v1.reshape(self.rnn, shape=[-1, h_size])
        if double_dueling == True:
            # The output from the recurrent player is then split into separate Value and Advantage streams
            self.streamA, self.streamV = tf.compat.v1.split(self.rnn, 2, 1)
            self.AW = tf.compat.v1.Variable(tf.compat.v1.random_normal([h_size // 2, num_action]))
            self.VW = tf.compat.v1.Variable(tf.compat.v1.random_normal([h_size // 2, 1]))
            self.Advantage = tf.compat.v1.matmul(self.streamA, self.AW)
            self.Value = tf.compat.v1.matmul(self.streamV, self.VW)

            self.salience = tf.compat.v1.gradients(self.Advantage, self.scalarInput)
            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.compat.v1.subtract(self.Advantage, tf.compat.v1.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        else:
            pass

        self.predict = tf.compat.v1.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.compat.v1.one_hot(self.actions, num_action, dtype=tf.float32)

        self.Q = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.compat.v1.square(self.targetQ - self.Q)

        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.compat.v1.zeros([self.batch_size, 7])
        self.maskB = tf.compat.v1.ones([self.batch_size, 3])
        self.mask = tf.compat.v1.concat([self.maskA, self.maskB], 1)
        self.mask = tf.compat.v1.reshape(self.mask, [-1])
        self.loss = tf.compat.v1.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class Agent:
    def __init__(self, replay_memory, beta, average, beta_2, average_2, num_actions_1, num_actions_2, input_shape,
                 batch_size=10, trace_length=TRACE_LENGTH, eps_evaluation=0.0, mu=0.0, tau=1.0, a=250.0, b=250.0, eps_constant=1.0):

        self.num_actions_1 = num_actions_1
        self.num_actions_2 = num_actions_2
        self.replay_memory = replay_memory
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.trace_length = trace_length

        # parameters of Epsilon-bmc
        self.mu0, self.tau0, self.a0, self.b0 = mu, tau, a, b
        self.stat = average
        self.post = beta
        self.eps_bmc = []
        self.eps_constant = eps_constant

        self.stat_2 = average_2
        self.post_2 = beta_2
        self.eps_bmc_2 = []

        self.eps_evaluation = eps_evaluation

    def calc_epsilon_1(self, eval=False):
        """
        Get the appropriate epsilon value for first agent
        """
        if eval == True:
            return self.eps_evaluation
        else:
            # expected value of Beta distribution (Adaptive eps-bmc)
            if len(self.replay_memory.action_1) > MIN_REPLAY_MEMORY_SIZE:
                post = self.post
                length_list = len(post.alpha)
                alpha = post.alpha[length_list-1]
                beta = post.beta[length_list-1]
                self.eps_bmc.append(alpha/(alpha + beta))
                return alpha / (alpha + beta)
            else:
                # Epsilon constant for the first phase: when buffer collects MEM_SIZE episodes, eps-bmc starts updating
                self.eps_bmc.append(self.eps_constant)
                return self.eps_constant

    def calc_epsilon_2(self, eval=False):
        """
        Get the appropriate epsilon value for second agent
        """
        if eval == True:
            return self.eps_evaluation
        else:
            # expected value of Beta distribution (Adaptive eps-bmc)
            if len(self.replay_memory.action_2) > MIN_REPLAY_MEMORY_SIZE:
                post = self.post_2
                length_list = len(post.alpha)
                alpha = post.alpha[length_list - 1]
                beta = post.beta[length_list - 1]
                self.eps_bmc_2.append(alpha / (alpha + beta))
                return alpha / (alpha + beta)
            else:
                # Epsilon constant for the first phase: when buffer collects MEM_SIZE episodes, eps-bmc starts updating
                self.eps_bmc_2.append(self.eps_constant)
                return self.eps_constant

    def update_posterior(self, data_1, data_2):
        length_list = len(self.post.alpha)
        alpha_1 = self.post.alpha[length_list - 1]
        beta_1 = self.post.beta[length_list - 1]
        alpha_2 = self.post_2.alpha[length_list - 1]
        beta_2 = self.post_2.beta[length_list - 1]
        # compute epsilon
        epsilon_1 = alpha_1 / (alpha_1 + beta_1)
        epsilon_2 = alpha_2 / (alpha_2 + beta_2)
        # (Credit to Michael Gimelfarb for this calculation)

        # update mu-hat and sigma^2-hat (sample mean and variance of the returns in D(set of previously observed returns))
        self.stat.update((1.0 - epsilon_1) * data_1[0] + epsilon_1 * data_1[1])
        self.stat_2.update((1.0 - epsilon_2) * data_2[0] + epsilon_2 * data_2[1])
        mu, t = self.stat.mean, self.stat.count
        last_var = len(self.stat.var)
        sigma2 = self.stat.var[last_var-1]
        mu_2, t_2 = self.stat_2.mean, self.stat_2.count
        sigma2_2 = self.stat_2.var[last_var - 1]

        # update a_t and b_t (parameters of marginal posterior distribution of the variance of the returns)
        a = self.a0 + t / 2
        a_2 = self.a0 + t_2 / 2
        b = self.b0 + t / 2 * sigma2 + t / 2 * (self.tau0 / (self.tau0 + t)) * (mu - self.mu0) * (mu - self.mu0)
        b_2 = self.b0 + t_2 / 2 * sigma2_2 + t_2 / 2 * (self.tau0 / (self.tau0 + t_2)) * (mu_2 - self.mu0) * (mu_2 - self.mu0)

        # compute e_t (under T-Student distribution)
        scale = (b / a) ** 0.5
        scale_2 = (b_2 / a_2) ** 0.5
        e_u_1 = tdist.pdf((1.0 - epsilon_1) * data_1[0] + epsilon_1 * data_1[1], df=2.0 * a, loc=data_1[1], scale=scale)
        e_q_1 = tdist.pdf((1.0 - epsilon_1) * data_1[0] + epsilon_1 * data_1[1], df=2.0 * a, loc=data_1[0], scale=scale)
        e_u_2 = tdist.pdf((1.0 - epsilon_2) * data_2[0] + epsilon_2 * data_2[1], df=2.0 * a_2, loc=data_2[1], scale=scale_2)
        e_q_2 = tdist.pdf((1.0 - epsilon_2) * data_2[0] + epsilon_2 * data_2[1], df=2.0 * a_2, loc=data_2[0], scale=scale_2)

        # update posterior
        self.post.update(e_u_1, e_q_1)
        self.post_2.update(e_u_2, e_q_2)


    def get_action(self, frame_number, main_drqn_steer, main_drqn_speed, state, state_in, state_in_2, session, eval=False):
        """
        Query the DRQN for an action given a state
        """
        eps_1 = self.calc_epsilon_1(eval)
        eps_2 = self.calc_epsilon_2(eval)

        if frame_number % 100000 == 0:
            #print("frame number: ", frame_number)
            #print("epsilon value: ", eps)
            pass

        # with chance epsilon, take a random choice
        if np.random.rand(1) < eps_1:
            # st_time = time.time()
            state1 = session.run(main_drqn_steer.rnn_state, feed_dict={main_drqn_steer.scalarInput: [state / 255.0], main_drqn_steer.trainLength: 1, main_drqn_steer.state_in: state_in, main_drqn_steer.batch_size: 1})
            action = np.random.randint(0, self.num_actions_1)
            time.sleep(1 / 25)
            action_1 = action
        else:
            action_1, state1 = session.run([main_drqn_steer.predict, main_drqn_steer.rnn_state], feed_dict={main_drqn_steer.scalarInput: [state / 255.0], main_drqn_steer.trainLength: 1, main_drqn_steer.state_in: state_in, main_drqn_steer.batch_size: 1})
            action_1 = action_1[0]

        if np.random.rand(1) < eps_2:
            # st_time = time.time()
            state2 = session.run(main_drqn_speed.rnn_state, feed_dict={main_drqn_speed.scalarInput: [state / 255.0], main_drqn_speed.trainLength: 1, main_drqn_speed.state_in: state_in_2, main_drqn_speed.batch_size: 1})
            action = np.random.randint(0, self.num_actions_2)
            time.sleep(1 / 25)
            action_2 = action
        else:
            action_2, state2 = session.run([main_drqn_speed.predict, main_drqn_speed.rnn_state], feed_dict={main_drqn_speed.scalarInput: [state / 255.0], main_drqn_speed.trainLength: 1, main_drqn_speed.state_in: state_in_2, main_drqn_speed.batch_size: 1})
            action_2 = action_2[0]
        return action_1, action_2, state1, state2

    # 'value' is used to calculate epsilon-BMC and VDBE
    def value(self, main_drqn_steer, main_drqn_speed,state, state_in, state_in_2, session):
        q_values_1 = session.run(main_drqn_steer.Qout, feed_dict={main_drqn_steer.scalarInput: [state / 255.0], main_drqn_steer.trainLength: 1, main_drqn_steer.state_in: state_in, main_drqn_steer.batch_size: 1})
        q_values_1 = q_values_1[0]
        q_values_2 = session.run(main_drqn_speed.Qout,
                                 feed_dict={main_drqn_speed.scalarInput: [state / 255.0],
                                            main_drqn_speed.trainLength: 1,
                                            main_drqn_speed.state_in: state_in_2,
                                            main_drqn_speed.batch_size: 1})
        q_values_2 = q_values_2[0]

        return q_values_1, q_values_2

    # add experience to buffer
    def add_experience(self, frames, actions_1, action_2, rewards_1, rewards_2, next_frames, terminal):
        self.replay_memory.add_experience(frames, actions_1, action_2, rewards_1, rewards_2, next_frames, terminal)

    def learn_1(self, main_drqn_steer, target_drqn_steer, batch_size, trace_length, gamma, state_train, session, frame_number):
        """
        First agent
        Sample a batch_size and use it to improve the DRQN.
        Returns the loss between the predicted and target Q as a float
        """
        if len(self.replay_memory.action_1) < batch_size:
            return
        # take sampled experience
        state, reward, action, next_state, terminal = self.replay_memory.sample_1(batch_size, trace_length)


        # main DQN estimates the best action in new states
        arg_q_max = session.run(main_drqn_steer.predict, feed_dict={main_drqn_steer.scalarInput: next_state/255.0, main_drqn_steer.trainLength: trace_length, main_drqn_steer.state_in: state_train, main_drqn_steer.batch_size: batch_size})

        # target DQN estimates the q values for new states
        future_q_values = session.run(target_drqn_steer.Qout, feed_dict={target_drqn_steer.scalarInput: next_state/255.0, target_drqn_steer.trainLength: trace_length, target_drqn_steer.state_in: state_train, target_drqn_steer.batch_size: batch_size})
        double_q = future_q_values[range(batch_size * trace_length), arg_q_max]

        # calculate targets with Bellman equation
        # if terminal_flags == 1 (the state is terminal), target_q is equal to rewards
        target_q = reward + (gamma * double_q * (1 - terminal))

        # use targets to calculate loss and use loss to calculate gradients
        loss, error, _ = session.run([main_drqn_steer.loss, main_drqn_steer.td_error, main_drqn_steer.updateModel], feed_dict={main_drqn_steer.scalarInput: state/255.0, main_drqn_steer.targetQ: target_q, main_drqn_steer.actions: action, main_drqn_steer.trainLength: trace_length, main_drqn_steer.state_in: state_train, main_drqn_steer.batch_size: batch_size})
        return float(loss), error

    def learn_2(self, main_drqn_speed, target_drqn_speed, batch_size, trace_length, gamma, state_train, session, frame_number):
        """
        Second agent
        Sample a batch_size and use it to improve the DRQN.
        Returns the loss between the predicted and target Q as a float
        """
        if len(self.replay_memory.action_2) < batch_size:
            return
        # take sampled experience
        state, reward, action, next_state, terminal = self.replay_memory.sample_2(batch_size, trace_length)


        # main DQN estimates the best action in new states
        arg_q_max = session.run(main_drqn_speed.predict, feed_dict={main_drqn_speed.scalarInput: next_state/255.0, main_drqn_speed.trainLength: trace_length, main_drqn_speed.state_in: state_train, main_drqn_speed.batch_size: batch_size})

        # target DQN estimates the q values for new states
        future_q_values = session.run(target_drqn_speed.Qout, feed_dict={target_drqn_speed.scalarInput: next_state/255.0, target_drqn_speed.trainLength: trace_length, target_drqn_speed.state_in: state_train, target_drqn_speed.batch_size: batch_size})
        double_q = future_q_values[range(batch_size * trace_length), arg_q_max]

        # calculate targets with Bellman equation
        # if terminal_flags == 1 (the state is terminal), target_q is equal to rewards
        target_q = reward + (gamma * double_q * (1 - terminal))

        # use targets to calculate loss and use loss to calculate gradients
        loss, error, _ = session.run([main_drqn_speed.loss, main_drqn_speed.td_error, main_drqn_speed.updateModel], feed_dict={main_drqn_speed.scalarInput: state/255.0, main_drqn_speed.targetQ: target_q, main_drqn_speed.actions: action, main_drqn_speed.trainLength: trace_length, main_drqn_speed.state_in: state_train, main_drqn_speed.batch_size: batch_size})
        return float(loss), error

    def save(self, folder_name, **kwargs):
        """
        Saves the Agent and all corresponding properties into a folder
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)


        # Save replay buffer
        self.replay_memory.save(folder_name + '/replay-memory')

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'alpha_1': self.post.alpha, 'alpha_2': self.post_2.alpha, 'beta_1': self.post.beta, 'beta_2': self.post_2.beta,
                                   'count_eps_1': self.stat.count, 'count_eps_2': self.stat_2.count, 'var_eps_1': self.stat.var,
                                   'var_eps_2': self.stat_2.var, 'm2_eps_1': self.stat.m2, 'm2_eps_2': self.stat_2.m2,
                                   'mean_eps_1': self.stat.mean, 'mean_eps_2': self.stat_2.mean,
                                   'epsilon_1': self.eps_bmc, 'epsilon_2': self.eps_bmc_2},
                                **kwargs}))  # save replay_memory information and any other information

    def load(self, folder_name, load_replay_memory=True):
        """Load a previously saved Agent from a folder
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')


        # Load replay buffer
        if load_replay_memory:
            self.replay_memory.load(folder_name + '/replay-memory')

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_memory:
            self.post.alpha = meta['alpha_1']
            self.post_2.alpha = meta['alpha_2']
            self.post.beta = meta['beta_1']
            self.post_2.beta = meta['beta_2']
            self.stat.count = meta['count_eps_1']
            self.stat_2.count = meta['count_eps_2']
            self.stat.m2 = meta['m2_eps_1']
            self.stat_2.m2 = meta['m2_eps_2']
            self.stat.var = meta['var_eps_1']
            self.stat_2.var = meta['var_eps_2']
            self.stat.mean = meta['mean_eps_1']
            self.stat_2.mean = meta['mean_eps_2']
            self.eps_bmc = meta['epsilon_1']
            self.eps_bmc_2 = meta['epsilon_2']
        return meta
