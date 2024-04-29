import numpy as np

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer

class LogPlusN(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, N=1):
        FuncTransformer.__init__(self, feature, new_feature, lambda x: np.log(N+x))
