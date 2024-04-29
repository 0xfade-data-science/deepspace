import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer

class Sum(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, features, new_feature, include_biais=False):
        FuncTransformer.__init__(self, None, new_feature, func = None)
        self.features = features
        self.include_biais = include_biais
    def apply(self):
        self.df[self.new_feature] = self.df[self.features].sum(axis=1)
        return self.df
    
class MinusTODO(Sum):
    '''Target Feature Engineering'''
    def __init__(self, features, new_feature, include_biais=False):
        Sum.__init__(self, features=features, new_feature=new_feature, include_biais=include_biais)
    def apply(self):
        self.df[self.new_feature] = self.df[self.features].sum(axis=1)
        return self.df
