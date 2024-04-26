import numpy as np
from DeepSpace.transformers.model.regression.linear.func.log.Abstract import FuncTransformer

class Exp(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
        FuncTransformer.__init__(self, feature, new_feature, np.exp)
    def transform_(self, ds):
        df = ds.test_data
        df[self.feature] = np.exp(df[self.new_feature])
        return ds
