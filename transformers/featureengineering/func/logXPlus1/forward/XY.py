import numpy as np

from deepspace.transformers.featureengineering.func.Forward import Forward

class LogXPlus1(Forward):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
        Forward.__init__(self, feature, new_feature, lambda x: np.log(x+1))