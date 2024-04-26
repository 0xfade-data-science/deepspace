import pandas as pd
from DeepSpace.transformers.model.regression.tree.performance.Show import Calc as PerformanceViewer
from DeepSpace.base import Base
from DeepSpace.DataSpace import DataSpace


class Show(PerformanceViewer):
    ''''''
    def __init__(self):
        PerformanceViewer.__init__(self)
    def transform(self, ds: DataSpace):
        pdf = pd.concat([ds.perf_train, ds.perf_test], keys=['train', 'test'], ignore_index=False)
        self.display(pdf)
        ds.perf_df = pdf
        return ds
