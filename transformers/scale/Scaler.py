import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

#from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
#from deepspace.transformers.outliers.Check import CheckOutliers

import deepspace.transformers as T 

class Scaler(T.Transformer.Transformer):
    '''not tested yet'''
    def __init__(self):
        T.Transformer.Transformer.__init__(self)
    def transform(self, ds:DataSpace):
        self.separator()
        self.df = self.scale(ds.data)
        ds.data = self.df
        ds.scaler = self.scaler
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

