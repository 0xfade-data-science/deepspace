import numpy as np

from deepspace.transformers.featureengineering.func.Backward import BackwardYTrain
    
class InvLogXPlus1(BackwardYTrain):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
            BackwardYTrain.__init__(self, feature, new_feature, lambda x: np.exp(x)-1)
