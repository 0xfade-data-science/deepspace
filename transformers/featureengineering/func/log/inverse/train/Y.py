import numpy as np

from deepspace.transformers.featureengineering.func.Backward import Backward

class Exp(Backward):
      '''Target Feature Engineering'''
      def __init__(self, feature, new_feature):
        Backward.__init__(self, new_feature, feature, np.exp)
      def init_from_ds(self, ds):
          self.ds = ds
          self.df = self.ds.inv_train_data_pred 
      def ds_init(self):
          self.ds.inv_train_data_pred = self.df
    
