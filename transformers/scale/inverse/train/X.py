import pandas as pd

#from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
#from deepspace.transformers.outliers.Check import CheckOutliers

import deepspace.transformers as T 
from deepspace.transformers.scale.Scaler2 import Scaler2

class InverseScaler2(Scaler2):
    '''not tested yet'''
    def __init__(self, num_cols=[]):
        Scaler2.__init__(self)
        self.num_cols = num_cols
    def transform(self, ds:DataSpace):
        self.separator()
        self.ds = ds
        self.scaler = ds.scaler
        self.num_cols = self.get_num_cols(ds.data)
        self.train_data = self.unscale()
        ds.isUnscaled = True
        ds.train_data = self.train_data
        return ds
    def unscale(self):
        #x_train = ds.x_train.copy()
        #if 'const' in ds.columns:
        #  x_train.drop(columns=['const'], inplace=True)
        #x_train[ds.target_col] = ds.y_train
        scaled_data = self.ds.train_data
        unscaled_data = self.scaler.inverse_transform(scaled_data)
        return pd.DataFrame(unscaled_data, columns=self.ds.data.columns, index=self.ds.x_train.index)
