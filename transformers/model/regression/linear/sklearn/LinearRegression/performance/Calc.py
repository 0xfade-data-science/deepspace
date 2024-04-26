
from DeepSpace.DataSpace import DataSpace
from DeepSpace.base import Base
#from DeepSpace.transformers.outliers.Check import CheckOutliers

from DeepSpace.transformers.model.regression.linear.statsmodel.performance.Calc import Calc as PerformanceCalculator

class Calc(PerformanceCalculator):
    def __init__(self) : 
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator.__init__(self)
    def predict(self, predictors):
        prediction = self.get_model().predict(predictors) # Predict
        return prediction
