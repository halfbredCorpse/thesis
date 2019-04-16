import tensorflow as tf
import numpy
from vizdoom import *

import random
import time
from skimage import

from py4j.java_gateway import get_field

class Dqn_1(object):
    def __init__(self, gateway):
        self.gateway = gateway

    def close(self):
        pass