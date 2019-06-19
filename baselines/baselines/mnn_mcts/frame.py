import numpy as np

from collections import deque
from skimage import transform

def preprocess_frame(frame):
    normalized_frame = frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    return preprocessed_frame

    stack_size = 4
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, new_episode):
    frame = preprocess_frame(state)

    if new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_frames, stacked_state

