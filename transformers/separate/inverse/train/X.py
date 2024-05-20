from deepspace.transformers.separate.Separator import Separator
from deepspace.DataSpace import DataSpace

class InverseSeparator(Separator):
    def __init__(self, target_col):
        Separator.__init__(self)
        if not target_col:
            raise Exception("target_col null")
        self.target_col = target_col
    def transform(self, ds: DataSpace):
        self.ds = ds
        self.sanity_check()
        ds = self.inverse_separate(ds)
        ds.isSeparateReverted_XTrain = True
        return ds
    def sanity_check(self):
        if not self.ds.isSeparateApplyed:
            raise Exception('not separated')
    def get_data(self):
        #if ds.isRmcUnApplyed_XTrain or ds.isRnsfUnApplyed_XTrain:
        if hasattr(self.ds, 'inv_train_data'):
            return self.ds.inv_train_data
        return self.ds.x_train.copy()
    def inverse_separate(self, ds: DataSpace):
        self.separator()
        #TODO optimize by avoiding copy
        data = self.get_data()
        data[self.target_col] = ds.y_train
        if 'const' in data:
            data.drop(columns=['const'], inplace=True)
        #take only new columns but reordered
        ordered_cols = self._get_ordered_cols(ds.data.columns, data.columns)
        #ordered_cols = ds.data.columns
        ds.inv_train_data = data[ordered_cols]

        ds.inverted_ds.x_train = ds.inv_train_data

        return ds
