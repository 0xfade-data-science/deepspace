
from deepspace.transformers.model.regression.linear.statsmodel.performance.Calc import Calc as PerformanceCalculator

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.base import Base

class Calc(PerformanceCalculator):
    def __init__(self) : 
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator.__init__(self)
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_train, ds.perf_test = self.calc_perf()
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds._model
    def predict(self, predictors):
        prediction = self.get_model().predict(predictors) # Predict
        #reshape for OLS to (rows, 1) (whose shape is (rows,) and creates memory problems) 
        #prediction = prediction.to_numpy().reshape(prediction.shape[0], 1)
        return prediction
