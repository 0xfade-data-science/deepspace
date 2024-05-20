
import pandas as pd

from deepspace.DataSpace import DataSpace
from deepspace.transformers.column.Abstract import Abstract

class ValueCounter(Abstract):
    def __init__(self, cat_cols = [], normalize=True, dropna=False, only=[], debug=False):
        super().__init__()
        self.normalize = normalize
        self.dropna = dropna
        self.cat_cols = cat_cols
        self.only = only
        self.debug = debug
    def transform(self, ds:DataSpace):
        cols = self._get_cat_cols(ds)
        self.show_count(ds.data, cols, normalize=self.normalize)
        return ds

    def show_count(self, df, cat_cols, normalize=True):
        self.separator()
        if len(cat_cols) <= 0:
            raise Exception('empty cat_cols')
        for column in cat_cols:
            print("-" * 50 + f" {column}")
            cntdf = pd.DataFrame(df[column].value_counts(normalize=False, dropna=self.dropna))
            pctdf = pd.DataFrame(df[column].value_counts(normalize=True, dropna=self.dropna))
            _df = pd.concat([cntdf, pctdf], axis=1)
            self.display(_df)
