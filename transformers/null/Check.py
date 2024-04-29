#from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
#from deepspace.transformers.outliers.Check import CheckOutliers

import deepspace.transformers.Transformer as T 

class CheckNulls(T.Transformer):
    ''''''
    def __init__(self, pct=True):
        T.Transformer.__init__(self)
        self.pct = pct
    def transform(self, ds: DataSpace):
        return self.check_nulls(ds)
    def check_nulls(self, ds: DataSpace):
        self.separator(caller=str(self))
        self._show_cols_with_null(ds.data, pct=self.pct)
        return ds
    def _show_cols_with_null(self, df, pct=True):
        if pct :
            self.print((df.isnull().sum() / df.shape[0])*100)
        else :
            self.print(df.isnull().sum())
