import pandas as pd

#from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
#from deepspace.transformers.outliers.Check import CheckOutliers

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
        self.inv_test_data = self.invert_scaling()
        ds.isUnscaled_XTest = True
        ds.inv_test_data = self.inv_test_data
        return ds
    def sanity_check(self, ds:DataSpace):
        if not ds.isScaled:
            raise Exception('not scaled')
        if ds.isUnscaled_XTest :
            raise Exception('already unscaled')
    def invert_scaling(self):
        self.separator()
        ordered_cols = self._get_ordered_cols(self.ds.cols_scaled, self.ds.inv_test_data)
        scaled_data = self.ds.inv_test_data[ordered_cols]
        unscaled_data = self.scaler.inverse_transform(scaled_data)
        return pd.DataFrame(unscaled_data, columns=self.ds.cols_scaled, index=self.ds.x_test.index)