import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer

class Polynom(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, features, new_feature,  n=2, include_biais=False):
        FuncTransformer.__init__(self, None, new_feature, func = None)
        self.features = features
        self.n = n
        self.include_biais = include_biais
    def apply(self):
        self.polynom_features = PolynomialFeatures(self.n, include_bias=self.include_biais)
        self.polynom_data = self.polynom_features.fit_transform(self.df[self.features])
        self.df_poly = pd.DataFrame(self.polynom_data, columns=self.polynom_features.get_feature_names_out())
        self.df[self.new_feature] = self.df_poly.sum(axis=1)
        return self.df
        
class PolynomOLD(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature1, feature2, new_feature,  n=1, include_biais=False):
        FuncTransformer.__init__(self, feature1, new_feature, func = None)
        self.feature1 = feature1
        self.feature2 = feature2
        self.include_biais = include_biais
        self.n = n
    def apply(self):
        self.polynom_features = PolynomialFeatures(self.n, include_bias=self.include_biais)
        self.polynom_data = self.polynom_features.fit_transform(self.df[[self.feature1, self.feature2]])
        self.df_poly = pd.DataFrame(self.polynom_data, columns=self.polynom_features.get_feature_names_out())
        self.df[self.new_feature] = self.df_poly.sum(axis=1)
        return self.df

