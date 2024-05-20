import numpy as np

from deepspace.transformers.featureengineering.func.Backward import BackwardXTrain

class InvLogXPlus1(BackwardXTrain):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
            BackwardXTrain.__init__(self, feature, new_feature, lambda x: np.exp(x)-1)
