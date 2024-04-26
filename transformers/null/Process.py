
from sklearn.impute import SimpleImputer

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
#from DeepSpace.transformers.outliers.Check import CheckOutliers

from DeepSpace.transformers.column.abstract import Abstract

class ProcessImputation(Abstract):
    ''' Only for X'''
    def __init__(self, num_cols=[], cat_cols=[], method='mean', drop_first=True):
        Abstract.__init__(self, num_cols=num_cols, cat_cols=cat_cols)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.method = method

    def transform(self, ds: DataSpace):
        self.ds = ds
        self.separator(caller=self)
        self.df = ds.data
        if self.method == 'drop':
            self.df = self.drop_nulls(ds)
        else :
            self.df = self.impute_nulls(ds)
        ds.data = self.df
        return ds
    def _show_cols_with_null(self, df, pct=True):
        #self.separator()
        if pct :
            self.print((df.isnull().sum() / df.shape[0])*100)
        else :
            self.print(df.isnull().sum())
    def _get_cols_with_null(self, ds):
        #self.separator()
        df = ds.data
        self._show_cols_with_null(df, pct=True)
        num_cols_to_impute = []
        cat_cols_to_impute = []
        num_cols = self._get_num_cols(ds)
        cat_cols = self._get_cat_cols(ds)
        for col, nbnuls in ds.data.isnull().sum().items():
            if nbnuls > 0:
                if col in num_cols:
                    num_cols_to_impute.append(col)
                elif col in cat_cols:
                    cat_cols_to_impute.append(col)
                #else:
                #    raise Exception(f'Unexpected column "{col}"')
        self.print("num cols to impute")
        self.print(num_cols_to_impute)
        self.print("cat cols to impute")
        self.print(cat_cols_to_impute)
        self.print("df cols")
        self.print(df.columns)
        return num_cols_to_impute, cat_cols_to_impute

    def drop_nulls(self, ds): # should be done before split or after ?
        self.separator()
        df = ds.data
        if self.method == 'drop':
            cols = self._get_num_cols(ds)
            for col in cols:
                idx = df[ df[ col ].isnull() ].index
                if idx.size > 0:
                    self.print(f'impute_nulls : removing {idx.size} rows for {col}')
                    df.drop(idx, axis="index", inplace=True)
        return df
    #imputes numerical by mean and categorical by mode
    def impute_nulls(self, ds): # should be done before split or after ?
        self.separator(caller=self)
        num_cols_to_impute, cat_cols_to_impute = self._get_cols_with_null(ds)
        df = ds.data
        # catagories
        if len(cat_cols_to_impute):
            self.separator(caller=self, string="imputing cat cols with 'mode'", nb=5, sep='-')
            imputer_mode = SimpleImputer(strategy="most_frequent")  # mode
            df[cat_cols_to_impute] = imputer_mode.fit_transform(df[cat_cols_to_impute])
        # numerical
        if len(num_cols_to_impute):
            self.separator(caller=self, string="imputing num cols with 'mean'", nb=5, sep='-')
            imputer_mode = SimpleImputer(strategy="mean")  # mean
            df[num_cols_to_impute] = imputer_mode.fit_transform(df[num_cols_to_impute])
        self._show_cols_with_null(df, pct=True)
        return df
