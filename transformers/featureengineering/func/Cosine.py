import numpy as np

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer

class Cosine(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
        FuncTransformer.__init__(self, feature, new_feature, np.cos)
