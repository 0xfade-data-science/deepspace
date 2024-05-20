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
        ds.isSeparateReverted_XTest = True        
        return ds
    def sanity_check(self):
        if not self.ds.isSeparateApplyed:
            raise Exception('not separated')
    def get_data(self):
        #if ds.isRmcUnApplyed_XTest or ds.isRnsfUnApplyed_XTest:
        if hasattr(self.ds, 'inv_test_data'):
            return self.ds.inv_test_data
        return self.ds.x_test.copy()

    def inverse_separate(self, ds: DataSpace):
        self.separator()

        data = self.get_data()
        data[self.target_col] = ds.y_test
        if 'const' in data:
            data.drop(columns=['const'], inplace=True)
        #take only new columns but reordered
        ordered_cols = self._get_ordered_cols(ds.data.columns, data.columns)
        ds.inv_test_data = data[ordered_cols]

        return ds