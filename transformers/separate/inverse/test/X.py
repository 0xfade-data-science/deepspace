from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace

class InverseSeparator(Transformer):
    def __init__(self, target_col):
        Transformer.__init__(self)
        if not target_col:
            raise Exception("target_col null")
        self.target_col = target_col
    def transform(self, ds: DataSpace):
        self.separator()
        df = ds.data
        data = ds.x_test.copy()
        if 'const' in data:
          data.drop(columns=['const'], inplace=True)
        data[self.target_col] = ds.y_test
        #take only new columns but reordered
        ordered_cols = ds.data.filter(items=data.columns).columns
        ds.test_data = data[ordered_cols]

        #data = data[ds.data.columns]
        #ds.test_data = data
        return ds