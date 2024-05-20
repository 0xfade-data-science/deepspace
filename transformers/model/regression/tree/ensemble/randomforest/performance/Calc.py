from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

from deepspace.transformers.model.regression.tree.performance.Calc import Calc as PerformanceCalculator
from deepspace.base import Base
from deepspace.DataSpace import DataSpace


class Calc(PerformanceCalculator):
    def __init__(self) :
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator.__init__(self)
    def transform_remove(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_train, ds.perf_test = self.calc_perf()
        return ds
    def from_ds_init_remove(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds._model
    def get_model_remove(self):
        return self._model
