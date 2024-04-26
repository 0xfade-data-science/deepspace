import numpy as np

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.featureengineering.func.Abstract import FuncTransformer

class LogPlusN(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, N=1):
        FuncTransformer.__init__(self, feature, new_feature, lambda x: np.log(N+x))
