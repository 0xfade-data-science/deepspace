from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace

class DropCols(Transformer):
    def __init__(self, cols=[], inplace=True):
        Transformer.__init__(self)
        self.cols = cols
        self.inplace = inplace
    def transform(self, ds: DataSpace):
        self.ds = ds
        df = self.drop(ds.data)
        self.adjust(df)
        return ds
    def adjust(self, df):
        self.ds.num_cols = self.minus_many(self.ds.num_cols, self.ds.drop_cols)
        self.ds.cat_cols = self.minus_many(self.ds.cat_cols, self.ds.drop_cols)
        self.ds.data = df
    def drop(self, df):
        if len(self.cols) <= 0:
            #intersept
            self.cols = self.ds.drop_cols
        self.cols = self.intercept(self.cols, df.columns.values.tolist())#only thos still present
        if self.inplace:
            df.drop(columns=self.cols, inplace=self.inplace)
        else:
            df = df.drop(columns=self.cols, inplace=self.inplace)
        self.print(f'droping cols ' + ','.join(self.cols))
        self.print(f'remaining cols ' + ','.join(df.columns.values.tolist()))
        #adjust num cols and cat cols
        return df
