import pandas as pd

from deepspace.transformers.exploration.plot.ScatterPlot import ScatterPlot  
from deepspace.DataSpace import DataSpace
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.exploration.plot.Abstract import AbstractPlot 

class TrainPredictionPlot(AbstractPlot):
    def __init__(self, target_col, figsize=(7, 7), xlabel='y1', ylabel='y2'):
        Transformer.__init__(self)
        self.target_col = target_col
        self.figsize = figsize
        self.xlabel=xlabel 
        self.ylabel=ylabel
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.plot()
        return ds
    def from_ds_init(self, ds):
        self.ds = ds
        self.x_train, self.y_train = ds.x_train, ds.y_train
        self.x_test, self.y_test = ds.x_test, ds.y_test
    def plot(self):
        df1, df2 = self.get_data()
        ScatterPlot(df1, df2, 
                        xlabel=self.xlabel, ylabel=self.ylabel, 
                        figsize=self.figsize)._plot(df1.values, df2.values)
    def get_data(self):
        df1 = self.ds.y_train[self.target_col]
        df2 = self.ds.y_train_pred[self.target_col]
        return df1, df2
        
class TestPredictionPlot(TrainPredictionPlot):
    def get_data(self):
        df1 = self.ds.y_test[self.target_col]
        df2 = self.ds.y_test_pred[self.target_col]
        return df1, df2
