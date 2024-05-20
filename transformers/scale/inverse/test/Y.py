import pandas as pd

from deepspace.DataSpace import DataSpace
from deepspace.transformers.scale.inverse.test.X import XInverseScaler2

class YInverseScaler2(XInverseScaler2):
    '''not tested yet'''
    def __init__(self, num_cols=[]):
        XInverseScaler2.__init__(self)
        self.num_cols = num_cols
    def transform(self, ds:DataSpace):
        self.sanity_check(ds)
        self.ds = ds
        self.scaler = ds.scaler
        self.num_cols = self.get_num_cols()
        self.inv_test_data_pred = self.invert_scaling()
        ds.isUnscaled_YTest = True
        ds.inv_test_data_pred = self.inv_test_data_pred
        return ds
    def sanity_check(self, ds:DataSpace):
        self.separator()
        if not ds.isScaled:
            raise Exception('not scaled')
        if ds.isUnscaled_YTest :
            raise Exception('already unscaled')
    def invert_scaling(self):
        self.separator()
        ordered_cols = self._get_ordered_cols(self.ds.cols_scaled, self.ds.inv_test_data_pred)
        scaled_data = self.ds.inv_test_data_pred[ordered_cols]
        unscaled_data = self.scaler.inverse_transform(scaled_data)
        return pd.DataFrame(unscaled_data, columns=self.ds.cols_scaled, index=self.ds.x_test.index)
