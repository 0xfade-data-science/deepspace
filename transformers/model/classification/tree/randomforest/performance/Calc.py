from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

from deepspace.transformers.model.classification.tree.performance.Calc import Calc as PerformanceCalculator
from deepspace.base import Base
from deepspace.DataSpace import DataSpace


class Calc(PerformanceCalculator):
    def __init__(self,threshold=0.5) :
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator.__init__(self, threshold=threshold)
    def get_prediction(self, model, predictors, kind='default'):
        return self._get_bin_prediction(model, predictors)
