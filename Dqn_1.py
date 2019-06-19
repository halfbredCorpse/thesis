import numpy
import warnings
import random
import retro

import tensorflow as tf
import matplotlib.pyplot as plot

from skimage import transform
from skimage.color import rgb2gray
from collections import deque

environment = retro.make(game="SpaceInvaders-Atari2600")
print("Size of frame: ", environment.observation_space)
print("Action size: ", environment.action_space.n)

possible_actions = numpy.array(numpy.identity(environment.action_space.n, dtype=int).tolist())


def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    return preprocessed_frame


STACK_SIZE = 4
stacked_frames = deque([numpy.zeros((110, 84), dtype=int) for i in range(STACK_SIZE)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([numpy.zeros((110, 84), dtype=int) for i in range(STACK_SIZE)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = numpy.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)

        stacked_state = numpy.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


state_size = [110, 84, 4]
action_size = environment.action_space.n
learning_rate = 0.000025

total_episodes = 50
max_steps = 50000
batch_size = 64

explore_start = 1.0
explore_stop = 0.1
decay_rate = 0.00001

gamma = 0.9

pretrain_length = batch_size
memory_size = 1000000

training = False
episode_render = False


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name="DQNetwork"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions = tf.placeholder(tf.float32, [None, self.action_size], name="actions")
            self.target_q = tf.placeholder(tf.float32, [None], name="target")
            self.convolution1 = tf.layers.conv2d(inputs=self.inputs,
                                                 filters=32,
                                                 kernel_size=[8, 8],
                                                 strides=[4, 4],
                                                 padding="VALID",
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                 name="convolution1")
            self.convolution1_out = tf.nn.elu(self.convolution1, name="convolution1_out")

            self.convolution2 = tf.layers.conv2d(inputs=self.convolution1_out,
                                                 filters=64,
                                                 kernel_size=[8, 8],
                                                 strides=[4, 4],
                                                 padding="VALID",
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                 name="convolution2")
            self.convolution2_out = tf.nn.elu(self.convolution2, name="convolution2_out")

            self.convolution3 = tf.layers.conv2d(inputs=self.convolution2_out,
                                                 filters=64,
                                                 kernel_size=[8, 8],
                                                 strides=[4, 4],
                                                 padding="VALID",
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                 name="convolution3")
            self.convolution3_out = tf.nn.elu(self.convolution3, name="convolution3_out")

            self.flatten = tf.contrib.layers.flatten(self.convolution3_out)
            self.final_convolution = tf.layers.dense(inputs=self.flatten,
                                                     units=512,
                                                     activation=tf.nn.elu,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                     name="fc1")

            self.output = tf.layers.dense(inputs=self.final_convolution,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          unit=self.action_size,
                                          activation=None)

            self.q_value = tf.reduce_sum(tf.multiply(self.output, self.actions))
            self.loss = tf.reduce_mean(tf.square(self.target_q - self.q_value))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


tf.reset_default_graph()
dqnetwork = DQNetwork(state_size, action_size, learning_rate)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = numpy.random.choice(numpy.arrange(buffer_size),
                                    size=batch_size,
                                    replace=False)

        return [self.buffer[i] for i in index]

memory = Memory(max_size=memory_size)

for i in range(pretrain_length):
    if i == 0:
        state = environment.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    choice = random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    next_state, reward, done, _ = environment.step(action)
    environment.render()

    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

    if done:
        next_state = numpy.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))
        state = environment.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        memory.add_experience((state, action, reward, next_state, done))
        state = next_state

writer = tf.summary.FileWriter("/tensorboard/dqn/1")
tf.summary.scalar("Loss", dqnetwork.loss)
write_op = tf.summary.merge_all()


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    explore_exploit_tradeoff = numpy.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * numpy.exp(-decay_rate * decay_rate)

    if explore_probability > explore_exploit_tradeoff:
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]
    else:
        q_s = sess.run(dqnetwork.output, feed_dict={dqnetwork.inputs : state.reshape((1, *state.shape))})
        choice = numpy.argmax(q_s)
        action = possible_actions[choice]

    return action, explore_probability


saver = tf.train.Saver()

if training:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer)
        decay_step = 0

        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            state = environment.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1
                decay_step += 1
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)
                next_state, reward, done, _ = environment.step(action)

                if episode_render:
                    environment.render()

                episode_rewards.append(reward)

                if done:
                    next_state = numpy.zeros((110, 84), dtype=numpy.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    step = max_steps
                    total_reward = numpy.sum(episode_rewards)

                    print('Episode: ', episode)
                    print('Total reward: ', total_reward)
                    print('Explore Probability: ', explore_probability)
                    print('Training Loss: ', loss)

                    #rewards_list.append((episode, total_reward))
                    memory.add_experience((state, action, reward, next_state, done))
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    memory.add_experience((state, action, reward, next_state, done))
                    state = next_state

                batch = memory.sample(batch_size)
                states_minibatch = numpy.array([each[0] for each in batch], ndmin=3)
                actions_minibatch = numpy.array([each[1] for each in batch])
                rewards_minibatch = numpy.array([each[2] for each in batch])
                next_states_minibatch = numpy.array([each[3] for each in batch], ndmin=3)
                dones_minibatch = numpy.array([each[4] for each in batch])

                target_q_s_batch = []
                q_s_next_state = sess.run(dqnetwork.output, feed_dict={dqnetwork.inputs : next_states_minibatch})

                for i in range(0, len(batch)):
                    terminal = dones_minibatch[i]

                    if terminal:
                        target_q_s_batch.append(rewards_minibatch[i])
                    else:
                        target = rewards_minibatch[i] + gamma * numpy.max(q_s_next_state[i])
                        target_q_s_batch.append(target)

                    targets_minibatch = numpy.array([each for each in target_q_s_batch])
                    loss, _ = sess.run([dqnetwork.loss, dqnetwork.optimizer],
                                       feed_dict={dqnetwork.inputs: states_minibatch,
                                                  dqnetwork.target_q: target_q_s_batch,
                                                  dqnetwork.actions: actions_minibatch})
                    summary = sess.run(write_op, feed_dict={dqnetwork.inputs: states_minibatch,
                                                            dqnetwork.target_q: target_q_s_batch,
                                                            dqnetwork.actions: actions_minibatch})

                    writer.add_summary(summary, episode)
                    writer.flush()

                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model saved")


with tf.Session as sess:
    total_test_rewards = []
    saver.restore(sess, "./models/model.ckpt")

    for episode in range(1):
        total_reward = 0
        state = environment.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        print("******************")
        print("Episode: ", episode)

        while True:
            state = state.reshape(1, *state_size)
            qs = sess.run(dqnetwork.output, feed_dict={dqnetwork.inputs: state})
            choice = numpy.argmax(qs)
            action = possible_actions[choice]
            next_state, reward, done, _ = environment.step(action)
            environment.render()
            total_reward += reward

            if done:
                print("Score: ", total_reward)
                total_test_rewards.append(total_reward)
                break

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state

    environment.close()