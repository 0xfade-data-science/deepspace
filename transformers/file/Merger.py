import pandas as pd
from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace


class Merger(Transformer):
    def __init__(self, on, ds1, ds2):
        super().__init__()
        self.on = on
        self.ds1, self.ds2 = ds1, ds2
    def transform(self, _):
        ds1, ds2 = self.ds1, self.ds2
        df1 = ds1.data
        df2 = ds2.data
        ds = ds1.clone()         
        ds.data = df1.merge(df2, on=self.on, copy=True)
        return ds