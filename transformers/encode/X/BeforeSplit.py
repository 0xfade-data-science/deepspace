import pandas as pd 

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.encode.Abstract import AbstractEncoder

class EncoderBeforeSplit(AbstractEncoder):
    def __init__(self, cat_cols=[], drop_first=True):
        AbstractEncoder.__init__(self, cat_cols=cat_cols)
        self.drop_first = drop_first
    def transform(self, ds: DataSpace):
        self.ds = ds
        self.df = ds.data
        self.separator(caller=str(self))
        self.cat_cols = self.get_cat_cols()
        self.num_cols = self.get_num_cols()
        self.df = ds.data.filter(items=self.cat_cols+self.num_cols)
        cols = self.get_cat_cols()
        self.print(f'-> considering cat cols = {", ".join(cols)}')
        self.df = pd.get_dummies(
            self.df,
            columns=cols,
            drop_first=self.drop_first,
        )
        self.df = self.df.astype(float) #prevents from bool in data types
        self.print(self.df.columns)
        ds.data = self.df
        return ds
