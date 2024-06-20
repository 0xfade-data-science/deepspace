import numpy as np

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

from deepspace.transformers.featureengineering.Abstract import FuncTransformer as AbstractFuncTransformer

class FuncTransformer(AbstractFuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, func):
        AbstractFuncTransformer.__init__(self, feature, new_feature)
        self.func = func
