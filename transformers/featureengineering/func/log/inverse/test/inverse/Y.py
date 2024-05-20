import numpy as np

from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer

class Exp(FuncTransformer):
      '''Target Feature Engineering'''
      def __init__(self, feature, new_feature):
        FuncTransformer.__init__(self, new_feature, feature, np.exp)
      def init_from_ds(self, ds):
          self.ds = ds
          self.df = self.ds.inv_test_data_pred 
      def ds_init(self):
          self.ds.inv_test_data_pred = self.df
    
