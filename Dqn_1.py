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
            self.convolution_1 = tf.layers.conv2d(inputs=self.inputs, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                                  padding="VALID")


