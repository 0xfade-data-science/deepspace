import numpy as np
from deepspace.transformers.model.regression.linear.func.log.Abstract import FuncTransformer

class Exp(FuncTransformer):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
        FuncTransformer.__init__(self, feature, new_feature, np.exp)
    def transform_(self, ds):
        df = ds.inv_test_data
        df[self.feature] = np.exp(df[self.new_feature])
        return ds
      def init_from_ds(self, ds):
            self.ds = ds
            self.df = self.ds.inv_train_data
      def ds_init(self):
            self.ds.inv_train_data = self.df          
