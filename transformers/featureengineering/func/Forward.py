import numpy as np

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer

class Forward(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, func):
        FuncTransformer.__init__(self, feature, new_feature, func)
    def init_from_ds(self, ds):
        self.ds = ds
        self.df = self.ds.data 
    def ds_init(self):
        self.ds.data = self.df