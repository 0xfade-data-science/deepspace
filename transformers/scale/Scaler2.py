import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.column.abstract import Abstract
#from deepspace.transformers.outliers.Check import CheckOutliers

import deepspace.transformers as T 

class Scaler2(Abstract):
    '''not tested yet'''
    def __init__(self, num_cols=[], exclude=[]):
        Abstract.__init__(self, num_cols=num_cols, exclude=exclude)
        #self.num_cols = num_cols
    def transform(self, ds:DataSpace):
        self.separator()
        self.ds = ds
        self.num_cols = self._get_num_cols(ds)
        self.df_num = ds.data.filter(items=self.num_cols, axis=1)
        self.df_num_scaled = self.scale(self.df_num)
        for col in self.num_cols :
            ds.data[col] = self.df_num_scaled[col]
        ds.scaler = self.scaler #store it so that we can unscale later
        ds.isScaled = True
        ds.isUnscaled = False
        return ds
    def scale(self, df): #x_train
        #Creating an instance of the MinMaxScaler
        self.scaler = MinMaxScaler()
        # Applying fit_transform on the training features data
        index = df.index
        columns = df.columns
        data = self.scaler.fit_transform(df)
        # The above scaler returns the data in array format, below we are converting it back to pandas DataFrame
        df = pd.DataFrame(data, index = index, columns = columns)
        return df
    def get_num_cols__NOTUSED(self, df):
        if len(self.num_cols) <= 0:
            if len(self.ds.num_cols) <=0:
                self.num_cols = self.org_num_cols = self.df.select_dtypes(
                    include=["number"]).columns.tolist()
            else:
                self.num_cols = self.ds.num_cols
        return self.num_cols
