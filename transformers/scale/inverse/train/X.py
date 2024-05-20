import pandas as pd

from deepspace.DataSpace import DataSpace

import deepspace.transformers as T 
from deepspace.transformers.scale.Scaler2 import Scaler2

class XInverseScaler2(Scaler2):
    '''not tested yet'''
    def __init__(self, num_cols=[]):
        Scaler2.__init__(self)
        self.num_cols = num_cols
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.scaler = ds.scaler
        self.num_cols = self.get_num_cols()
        self.sanity_check()        
        self.inv_train_data = self.invert_scaling()
        ds.isUnscaled_XTrain = True
        ds.inv_train_data = self.inv_train_data
        return ds
    def sanity_check(self):
        self.separator()
        if not self.ds.isScaled:
            raise Exception('not scaled')
        if self.ds.isUnscaled_XTrain :
            raise Exception('already unscaled')
    def invert_scaling(self):
        self.separator()
        ordered_cols = self._get_ordered_cols(self.ds.cols_scaled, self.ds.inv_train_data)
        scaled_data = self.ds.inv_train_data[ordered_cols]
        unscaled_data = self.scaler.inverse_transform(scaled_data)
        return pd.DataFrame(unscaled_data, columns=self.ds.cols_scaled, index=scaled_data.index)
