import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from deepspace.transformers.model.regression.linear.statsmodel.performance.Calc import Calc as PerformanceCalculator
from deepspace.DataSpace import DataSpace
from deepspace.base import Base
    
class Calc(PerformanceCalculator):
    def __init__(self, target_col) :
        PerformanceCalculator.__init__(self)
        self.target_col = target_col
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_test = self.perf_test()
        self.display(ds.perf_test)
        return ds
    def perf_test(self):
        self.separator()
        #self.test_rmse = np.sqrt(mean_squared_error(self.ds.test_data[self.target_col],
        #                                            self.ds.test_data_pred[self.target_col]))
        #self.display(pd.DataFrame(data={'test_rmse' : [self.test_rmse]}))
        if 'const' in self.ds.x_test: # for sklearn which does not expose 'const'
            self.ds.test_data['const'] = self.ds.x_test['const']
        self.perf_df = self.performance(self.ds.test_data[self.ds.x_test.columns],
                                        self.ds.test_data[self.target_col])
        return self.perf_df
    def performance(self, predictors, target):
        #prediction = self.predict(predictors) # Predict
        prediction = self.ds.test_data_pred[self.target_col]
        # dataframe of metrics to make it easy to view, compere and work with
        df_perf = self.as_dataframe(predictors, prediction, target)
        return df_perf