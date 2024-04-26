from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.outliers.Check import CheckOutliers

class ProcessOutliers(CheckOutliers):
    ''' Only for Y, not tested yet'''
    def __init__(self, num_cols=[], method='drop', factor=1.5):
        CheckOutliers.__init__(self, num_cols=num_cols, factor=factor)        
        self.method = method
    def transform(self, ds: DataSpace):
        cols, lower_bound, upper_bound = self.get_bounderies(ds)
        if self.method == 'drop':
            for col in cols:
                if len(self.num_cols) <=0 or col in self.num_cols :
                    idx = ds.data[ ds.data[ col ] < lower_bound[ col ] ].index
                    if idx.size > 0:
                        self.print(f'lower_bound: removing outliers {idx.size} rows for {col}')
                        ds.data.drop(idx, axis="index", inplace=True)
                    idx = ds.data[ ds.data[ col ] > upper_bound[ col ] ].index
                    if idx.size > 0:
                        self.print(f'upper_bound: removing outliers {idx.size} rows for {col}')
                        ds.data.drop(idx, axis="index", inplace=True)
        elif self.method == 'median': #never take mean
            median = ds.data.median(numeric_only=True)
            for col in cols:
                if len(self.num_cols) <=0 or col in self.num_cols :
                    idx = ds.data[ ds.data[ col ] > upper_bound[ col ] ].index
                    if idx.size > 0:
                        self.print(f'averaging outliers {idx.size} rows for {col}')
                        ds.data.loc[idx, col] = median[col]
                    idx = ds.data[ ds.data[ col ] < lower_bound[ col ] ].index
                    if idx.size > 0:
                        self.print(f'averaging outliers {idx.size} rows for {col}')
                        ds.data.loc[idx, col] = median[col]
        return ds
