import pandas as pd
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

from deepspace.transformers.model.regression.tree.performance.Show import Show as PerformanceViewer
from deepspace.base import Base
from deepspace.DataSpace import DataSpace

class Show(PerformanceViewer):
    ''''''
    def __init__(self):
        PerformanceViewer.__init__(self)

