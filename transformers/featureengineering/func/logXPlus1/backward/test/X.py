import numpy as np

from deepspace.transformers.featureengineering.func.Backward import BackwardXTest

class InvLogXPlus1(BackwardXTest):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
            BackwardXTest.__init__(self, feature, new_feature, lambda x: np.exp(x)-1)
