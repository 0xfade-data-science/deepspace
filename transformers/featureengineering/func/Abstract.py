import numpy as np

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

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
        self.separator(caller=self, string=f'applying function : {self.new_feature} = {self.func}({self.feature})')
        self.df[self.new_feature] = self.func(self.df[self.feature])
        return self.df
    def init_from_ds(self, ds):
        #self.ds = ds
        #self.df = self.ds.data 
        raise Exception('unexpected in abstract class')
    def ds_init(self):
        #self.ds.data = self.df
        raise Exception('unexpected in abstract class')
