from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class CheckUniqueness(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        self.check_unique(ds.data)
        return ds
    def get_uniqueness(self, df):
        self.separator()
        return df.nunique()
    def check_unique(self, df):
        self.separator()
        self.view_uniqueness(df)
    def view_uniqueness(self, df):
        self.separator()
        self.display(self.get_uniqueness(df).to_frame())
