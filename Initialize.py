import random
import warnings
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as kerasBackend


class Initialize():
    seed = 1
    def __init__(self, seed=1, ignorewarns=True):
        Initialize.seed = seed
        #self.seed = seed
        self.ignorewarns = ignorewarns
        print(f"tensorflow version {tf.__version__}")
        kerasBackend.clear_session()
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        # To ignore warnings
        if ignorewarns:
            warnings.filterwarnings("ignore")
