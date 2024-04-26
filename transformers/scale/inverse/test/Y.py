import pandas as pd

#from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
#from DeepSpace.transformers.outliers.Check import CheckOutliers

import DeepSpace.transformers as T 
from DeepSpace.transformers.scale.inverse.test.X import XInverseScaler2

class YInverseScaler2(XInverseScaler2):
    '''not tested yet'''
    def __init__(self, num_cols=[]):
        Transformer.__init__(self)
        self.num_cols = num_cols
    def transform(self, ds:DataSpace):
        self.separator()
        self.ds = ds
        self.scaler = ds.scaler
        self.num_cols = self.get_num_cols(ds.data)
        self.test_data_pred = self.unscale()
        ds.isUnscaled = True
        ds.test_data_pred = self.test_data_pred
        return ds
    def unscale(self):
        #x_test = ds.x_test.copy()
        #if 'const' in ds.columns:
        #  x_test.drop(columns=['const'], inplace=True)
        #x_test[ds.target_col] = ds.y_test
        scaled_data = self.ds.test_data_pred
        unscaled_data = self.scaler.inverse_transform(scaled_data)
        return pd.DataFrame(unscaled_data, columns=self.ds.data.columns, index=self.ds.x_test.index)
