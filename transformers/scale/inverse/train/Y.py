import pandas as pd

#from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
#from deepspace.transformers.outliers.Check import CheckOutliers

import deepspace.transformers as T 
from deepspace.transformers.scale.Scaler2 import Scaler2
from deepspace.transformers.scale.inverse.train.X import XInverseScaler2

class YInverseScaler2(XInverseScaler2):
    '''not tested yet'''
    def __init__(self, num_cols=[]):
        XInverseScaler2.__init__(self)
        self.num_cols = num_cols
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.scaler = ds.scaler
        self.num_cols = self.get_num_cols()
        self.inv_train_data_pred = self.invert_scaling()
        ds.isUnscaled_YTrain = True
        ds.inv_train_data_pred = self.inv_train_data_pred
        return ds
    def sanity_check(self, ds:DataSpace):
        self.separator()
        if not ds.isScaled:
            raise Exception('not scaled')
        if ds.isUnscaled_YTrain :
            raise Exception('already unscaled')
    def invert_scaling(self):
        self.separator()
        ordered_cols = self._get_ordered_cols(self.ds.cols_scaled, self.ds.inv_train_data_pred)
        scaled_data = self.ds.inv_train_data_pred[ordered_cols]
        unscaled_data = self.scaler.inverse_transform(scaled_data)
        return pd.DataFrame(unscaled_data, columns=self.ds.cols_scaled, index=scaled_data.index)
