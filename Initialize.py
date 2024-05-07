import sys
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as kerasBackend
import matplotlib
import sklearn
import statsmodels

from deepspace.base import Base 

class Initialize(Base):
    seed = 1
    ignorewarns = False
    matplotlib_inline = True
    def __init__(self, seed=1, matplotlib_inline=True, ignorewarns=True):
        Initialize.seed = seed
        Initialize.matplotlib_inline = matplotlib_inline
        Initialize.ignorewarns = ignorewarns
        kerasBackend.clear_session()
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        #set backend
        if matplotlib_inline:
            matplotlib.use('module://matplotlib_inline.backend_inline') #same as %matplotlib inline
        # To ignore warnings
        if ignorewarns:
            warnings.filterwarnings("ignore")
        self.versions()

    def versions(self):
        print(30*"#")
        (major, minor, micro, releaselevel, serial) = sys.version_info
        pyver = f'{major}.{minor}.{micro}'
        df = pd.DataFrame([pyver, pd.__version__, np.__version__, tf.__version__, sklearn.__version__, statsmodels.__version__], 
                          index=['python', 'pandas', 'numpy', 'tensorflow', 'sklearn', 'statsmodels'],
                          columns=['version']) 
        df = pd.DataFrame([pd.__version__, np.__version__, tf.__version__], columns=['version'], index=['pandas', 'numpy', 'tensorflow'])
        self.display(df)
        print(30*"#")
