from deepspace.transformers.drop.DropCols import DropCols
from deepspace.DataSpace import DataSpace

class DropColsXYXY(DropCols):
    def __init__(self, cols=[], inplace=True):
        DropCols.__init__(self, cols=cols, inplace=inplace)
    def transform(self, ds: DataSpace):
        self.ds = ds
        self.ds.x_train = self.drop(ds.x_train)
        self.ds.x_test = self.drop(ds.x_test)
        self.adjust()
        return ds
    def adjust(self):
        self.ds.num_cols = self.minus_many(self.ds.num_cols, self.ds.drop_cols)
        self.ds.cat_cols = self.minus_many(self.ds.cat_cols, self.ds.drop_cols)
