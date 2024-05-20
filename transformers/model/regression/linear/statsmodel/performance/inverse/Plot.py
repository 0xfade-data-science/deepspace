import pandas as pd

from deepspace.transformers.exploration.plot.ScatterPlot import ScatterPlot  
from deepspace.DataSpace import DataSpace
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.model.regression.linear.statsmodel.performance.Plot import TrainPredictionPlot as ForwardTrainPredictionPlot

class TrainPredictionPlot(ForwardTrainPredictionPlot):
    def get_data(self):
        df1 = self.ds.inv_train_data[self.target_col]
        df2 = self.ds.inv_train_data_pred[self.target_col]
        return df1, df2
        
class TestPredictionPlot(TrainPredictionPlot):
    def get_data(self):
        df1 = self.ds.inv_test_data[self.target_col]
        df2 = self.ds.inv_test_data_pred[self.target_col]
        return df1, df2
