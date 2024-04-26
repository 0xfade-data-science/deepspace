import numpy as np
from DeepSpace.transformers.featureengineering.func.Abstract import FuncTransformer

class Sqrt(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
        FuncTransformer.__init__(self, feature, new_feature, np.sqrt)
