import numpy as np

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer

class Backward(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, func):
        FuncTransformer.__init__(self, new_feature, feature, func)
    def init_from_ds(self, ds):
        raise Exception('unexpected in abstract class')
    def ds_init(self):
        raise Exception('unexpected in abstract class')

class BackwardXTrain(Backward):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, func):
        Backward.__init__(self, feature, new_feature, func)
    def init_from_ds(self, ds):
        self.ds = ds
        self.df = self.ds.inv_train_data
    def ds_init(self):
        self.ds.data = self.df

class BackwardYTrain(Backward):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, func):
        Backward.__init__(self, feature, new_feature, func)
    def init_from_ds(self, ds):
        self.ds = ds
        self.df = self.ds.inv_train_data_pred
    def ds_init(self):
        self.ds.data = self.df

class BackwardXTest(Backward):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, func):
        Backward.__init__(self, feature, new_feature, func)
    def init_from_ds(self, ds):
        self.ds = ds
        self.df = self.ds.inv_test_data 
    def ds_init(self):
        self.ds.data = self.df

class BackwardYTest(Backward):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, func):
        Backward.__init__(self, feature, new_feature, func)
    def init_from_ds(self, ds):
        self.ds = ds
        self.df = self.ds.inv_test_data_pred
    def ds_init(self):
        self.ds.data = self.df