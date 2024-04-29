from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.encode.X.BeforeSplit import EncoderBeforeSplit

from deepspace.transformers.column.abstract import Abstract

class CheckOutliers(Abstract):
    ''' Only for Y, not tested yet'''
    def __init__(self, num_cols=[], factor=1.5, show=False):
        Abstract.__init__(self, num_cols=num_cols)
        self.num_cols = num_cols
        self.factor= factor
        self.show = show
    def transform(self, ds: DataSpace):
        return self.check_outliers(ds)
    def check_outliers(self, ds: DataSpace):
        self.separator(caller=str(self))
        cols, lower_bound, upper_bound = self.get_bounderies(ds)
        for col in cols:
            if len(self.num_cols) <=0 or col in self.num_cols :
                idx = ds.data[ ds.data[ col ] < lower_bound[ col ] ].index
                if idx.size > 0:
                    #self.print(f'Feature *{col}*: found {idx.size} lower_bound outliers')
                    self.process_lower_bound(ds, col, idx)
                idx = ds.data[ ds.data[ col ] > upper_bound[ col ] ].index
                if idx.size > 0:
                    #self.print(f'Feature *{col}*: found {idx.size} upper_bound rows outliers')
                    self.process_upper_bound(ds, col, idx)
        return ds
    def get_bounderies(self, ds):
        #pdb.set_trace()
        q1 = ds.data.quantile(0.25, numeric_only=True)
        q3 = ds.data.quantile(0.75, numeric_only=True)
        iqr = q3-q1
        lower_bound = q1 - (self.factor*iqr)
        upper_bound = q3 + (self.factor*iqr)
        cols = q1.index.to_list()
        return cols, lower_bound, upper_bound
    def process_lower_bound(self, ds, col, idx):
          self.print(f'Feature *{col}*: found {idx.size} lower_bound outliers')
          if self.show:
            self.print(ds.data[col].loc[idx])
    def process_upper_bound(self, ds, col, idx):
          self.print(f'Feature *{col}*: found {idx.size} upper_bound rows outliers')
          if self.show:
            self.print(ds.data[col].loc[idx])


