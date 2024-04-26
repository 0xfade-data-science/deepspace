
from DeepSpace.transformers.model.regression.linear.statsmodel.performance.Calc import Calc as PerformanceCalculator

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
from DeepSpace.base import Base

class Calc(PerformanceCalculator):
    def __init__(self) : #, perfchecker : MyPerformanceChecker):
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator.__init__(self)
    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.perf_train, ds.perf_test = self.calc_perf()
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self._model = ds._model