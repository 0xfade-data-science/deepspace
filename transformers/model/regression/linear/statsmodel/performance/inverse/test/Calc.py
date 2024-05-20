from deepspace.transformers.model.regression.linear.statsmodel.performance.Calc import Calc as PerformanceCalculator
from deepspace.DataSpace import DataSpace
from deepspace.base import Base
    
class Calc(PerformanceCalculator):
    def __init__(self, target_col, doview=True):
        PerformanceCalculator.__init__(self)
        self.target_col = target_col
        self.doview = doview
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        self.calc_perf()
        self.ds.perf_test = self.perf_df
        self.show()
        return ds
    def calc_perf(self):
        self.separator()
        if 'const' in self.ds.x_test: # for sklearn which does not expose 'const'
            self.ds.inv_test_data['const'] = self.ds.x_test['const']
        X = self.ds.inv_test_data[self.ds.x_test.columns]
        Y = self.ds.inv_test_data[self.target_col]
        self.perf_df = self.performance(X, Y)
        return self.perf_df
    def performance(self, predictors, target):
        #prediction = self.predict(predictors) # Predict
        prediction = self.ds.inv_test_data_pred[self.target_col]
        # dataframe of metrics to make it easy to view, compere and work with
        df_perf = self.as_dataframe(predictors, prediction, target)
        return df_perf
    def show(self):
        if self.doview:
            self.display(self.ds.perf_test)
