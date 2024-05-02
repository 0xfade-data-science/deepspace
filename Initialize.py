import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as kerasBackend

from deepspace.base import Base 

class Initialize(Base):
    seed = 1
    def __init__(self, seed=1, ignorewarns=True):
        Initialize.seed = seed
        #self.seed = seed
        self.ignorewarns = ignorewarns
        kerasBackend.clear_session()
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        # To ignore warnings
        if ignorewarns:
            warnings.filterwarnings("ignore")
        self.versions()

    def versions(self):
        print(30*"#")
        df = pd.DataFrame([pd.__version__, np.__version__, tf.__version__], columns=['version'], index=['pandas', 'numpy', 'tensorflow'])
        self.display(df)
        print(30*"#")
