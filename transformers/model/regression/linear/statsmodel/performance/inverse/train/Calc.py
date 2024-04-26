
from DeepSpace.transformers.model.regression.linear.statsmodel.performance.Calc import Calc as PerformanceCalculator
from DeepSpace.DataSpace import DataSpace
from DeepSpace.base import Base

class Calc(PerformanceCalculator):
    def __init__(self, target_col) :
        PerformanceCalculator.__init__(self)
        self.target_col = target_col
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_train = self.perf_train()
        self.display(ds.perf_train)
        return ds
    def perf_train(self):
        self.separator()
        #self.train_rmse = np.sqrt(mean_squared_error(self.ds.train_data[self.target_col],
        #                                             self.ds.train_data_pred[self.target_col]))
        #self.display(pd.DataFrame(data={'train_rmse' : [self.train_rmse]}))
        if 'const' in self.ds.x_train: # sklearn linearmodel does not expose 'const', it uses it internaly only
            self.ds.train_data['const'] = self.ds.x_train['const']
        self.perf_df = self.performance(self.ds.train_data[self.ds.x_train.columns],
                                        self.ds.train_data[self.target_col])
        return self.perf_df
    def performance(self, predictors, target):
        #pdb.set_trace()
        #prediction = self.predict(predictors) # Predict
        prediction = self.ds.train_data_pred[self.target_col]
        # dataframe of metrics to make it easy to view, compere and work with
        df_perf = self.as_dataframe(predictors, prediction, target)
        return df_perf
    