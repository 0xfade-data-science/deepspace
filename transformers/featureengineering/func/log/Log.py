import numpy as np

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.featureengineering.func.Abstract import FuncTransformer

class Log(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
        FuncTransformer.__init__(self, feature, new_feature, np.log)
