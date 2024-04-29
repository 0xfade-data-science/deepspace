import numpy as np

from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer

class Exp(FuncTransformer):
      '''Target Feature Engineering'''
      def __init__(self, feature, new_feature):
            FuncTransformer.__init__(self, feature, new_feature, np.exp)
      def transform_NOTUSED(self, ds):
          df = ds.train_data
          df[self.feature] = np.exp(df[self.new_feature])
          return ds
      def init_from_ds(self, ds):
            self.ds = ds
            self.df = self.ds.test_data
      def ds_init(self):
            self.ds.test_data = self.df      