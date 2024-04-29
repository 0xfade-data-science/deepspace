import numpy as np

from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer

class Exp(FuncTransformer):
      '''Target Feature Engineering'''
      def __init__(self, feature, new_feature):
        FuncTransformer.__init__(self, feature, new_feature, np.exp)
        self.feature = feature
        self.new_feature = new_feature
      def transformOLD(self, ds):
        df = ds.train_data_pred
        df[self.feature] = np.exp(df[self.new_feature])
        return ds      
      def init_from_ds(self, ds):
          self.ds = ds
          self.df = self.ds.train_data_pred 
      def ds_init(self):
          self.ds.train_data_pred = self.df
    
