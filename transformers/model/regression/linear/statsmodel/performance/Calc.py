import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.base import Base

class Calc(Transformer):
    def __init__(self) : 
        Base.__init__(self, '#!#', 50)
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_train, ds.perf_test = self.calc_perf()                
        return ds
    def from_ds_init(self, ds):
        self.ds = ds
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds.get_model()
    def get_model(self):
        return self._model
    def calc_perf(self):
        self.separator()
        return self.perf_train(), self.perf_test()
    def perf_train(self):
        self.separator()
        self.perf_train_vals, self.ds.y_train_pred = self.performance(self.x_train, self.y_train)
        return self.perf_train_vals
    def perf_test(self):
        self.separator()
        self.perf_test_vals, self.ds.y_test_pred  = self.performance(self.x_test, self.y_test)
        return self.perf_test_vals
    def get_perf(self):
        return self.perf_train_vals, self.perf_test_vals
    def predict(self, predictors):
        prediction = self.get_model().predict(predictors) # Predict
        #reshape for OLS to (rows, 1) (whose shape is (rows,) and creates memory problems) 
        prediction = prediction.to_numpy().reshape(prediction.shape[0], 1)
        return prediction
    # Assemble metrics for our regression model
    def performance(self, predictors, target):
        prediction = self.predict(predictors) # Predict
        # dataframe of metrics to make it easy to view, compere and work with
        df_perf = self.as_dataframe(predictors, prediction, target)
        return df_perf, prediction
    def get_rmse(self, target, prediction):
        return np.sqrt(mean_squared_error(target, prediction))
    def get_mae(self, target, prediction):
        return mean_absolute_error(target, prediction)
    def get_mape(self, target, prediction):
        return np.mean(np.abs(target - prediction) / target) * 100
    def get_r2(self, target, prediction):
        return r2_score(target, prediction)
    def adj_r2(self, predictors, targets, predictions):
        r2 = r2_score(targets, predictions)
        n = predictors.shape[0]
        k = predictors.shape[1]
        return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

    def as_dataframe(self, predictors, prediction, target):
        return pd.DataFrame({
                            "RMSE":             self.get_rmse(target, prediction), #Root of Mean Squared Error (RMSE)
                            "MAE":              self.get_mae(target, prediction),          #Mean Absolute Error (MAE)
                            "MAPE":             self.get_mape(target, prediction),             #Mean Absolute Percent Error (MAPE)
                            "R-squared":        self.get_r2(target, prediction),
                            "Adj. R-squared":   self.adj_r2(predictors, target, prediction),
                        },
                        index=[0]
                      )
