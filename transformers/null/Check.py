import pandas as pd

#from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
#from deepspace.transformers.outliers.Check import CheckOutliers

import deepspace.transformers.Transformer as T 

class CheckNulls(T.Transformer):
    '''Shows count and percentage of null values'''
    def __init__(self):
        T.Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        return self.check_nulls(ds)
    def check_nulls(self, ds: DataSpace):
        self.separator(caller=str(self))
        self._show_cols_with_null_detailed(ds.data)
        return ds
    def _show_cols_with_null_detailed(self, df):
        #TODO make it one row with sum and percentage
        dfsum = pd.DataFrame(df.isnull().sum()).rename(columns={0: 'count'})
        dfpct = pd.DataFrame((df.isnull().sum() / df.shape[0])*100).rename(columns={0: 'pct'})
        _df = pd.concat([dfsum, dfpct], axis=1)
        self.display(_df)

class CheckNullsOLD(T.Transformer):
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
        #TODO make it one row with sum and percentage
        if pct :
            self.print((df.isnull().sum() / df.shape[0])*100)
        else :
            self.print(df.isnull().sum())
    def _show_cols_with_null_detailed(self, df, pct=True):
        #TODO make it one row with sum and percentage
        dfpct = (df.isnull().sum() / df.shape[0])*100
        dfsum = df.isnull().sum()
        _df = pd.DataFrame([dfsum, dfpct], columns=dfsum.columns, axis=0)
        self.display()
