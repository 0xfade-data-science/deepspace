import numpy as np

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace

class FuncTransformer(Transformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, func):
        Transformer.__init__(self)
        self.feature = feature
        self.new_feature = new_feature
        self.func = func
    def transform(self, ds:DataSpace):
        self.init_from_ds(ds)
        self.apply()
        self.ds_init()
        return self.ds
    def apply(self):
        self.df[self.new_feature] = self.func(self.df[self.feature])
        return self.df
    def init_from_ds(self, ds):
        self.ds = ds
        self.df = self.ds.data 
    def ds_init(self):
        self.ds.data = self.df