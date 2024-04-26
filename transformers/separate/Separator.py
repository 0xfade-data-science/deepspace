import pandas as pd 

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.column.abstract import Abstract

class Separator(Abstract):
    def __init__(self, target_col=None):
        Transformer.__init__(self)
        self.target_col = target_col
    def transform(self, ds: DataSpace):
        self.separator(caller=self)
        df = ds.data
        self.target_col = self._get_target_col(ds)
        x = df.drop(self.target_col, axis=1)
        y = df[self.target_col]
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        self.display(x.head())
        self.display(y.head())
        ds.x, ds.y = x,y
        self.display(x.head())
        self.display(y.head())
        return ds
