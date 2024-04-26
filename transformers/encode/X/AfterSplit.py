import pandas as pd

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.encode.X.BeforeSplit import EncoderBeforeSplit

class EncoderAfterSplit(EncoderBeforeSplit):
    def __init__(self, target_col, cat_cols=[], drop_first=True):
        EncoderBeforeSplit.__init__(self, cat_cols=cat_cols)
        self.target_col = target_col
        self.drop_first = drop_first
    def transform(self, ds):
        self.separator(caller=self)
        self.df = ds.x # for get_cols
        cols = self.get_cat_cols()
        self.print(f'-> considering cat cols = {", ".join(cols)}')
        ds.x = pd.get_dummies(ds.x, columns=cols, drop_first=self.drop_first)
        ds.y = pd.get_dummies(ds.y, columns=[self.target_col], drop_first=self.drop_first)
        return ds
