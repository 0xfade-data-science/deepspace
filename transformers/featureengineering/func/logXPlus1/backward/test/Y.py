import numpy as np

from deepspace.transformers.featureengineering.func.Backward import BackwardYTest
    
class InvLogXPlus1(BackwardYTest):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
            BackwardYTest.__init__(self, feature, new_feature, lambda x: np.exp(x)-1)
