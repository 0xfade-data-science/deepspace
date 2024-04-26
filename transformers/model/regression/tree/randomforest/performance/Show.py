import pandas as pd
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

from DeepSpace.transformers.model.regression.tree.performance.Show import Show as PerformanceViewer
from DeepSpace.base import Base
from DeepSpace.DataSpace import DataSpace

class Show(PerformanceViewer):
    ''''''
    def __init__(self):
        PerformanceViewer.__init__(self)

