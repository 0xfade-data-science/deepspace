import pandas as pd


from DeepSpace.base import Base
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.model.classification.logistic.sklearn.LogisticRegression.performance.Calc import Calc as PerformanceCalculator

class Calc(PerformanceCalculator):
    def __init__(self, threshold=0.5) : 
        Base.__init__(self, '#!#', 50)
        PerformanceCalculator .__init__(self, threshold=threshold)



