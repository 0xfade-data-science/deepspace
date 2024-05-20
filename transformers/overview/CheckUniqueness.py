from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class CheckUniqueness(Transformer):
    ''''''
    def __init__(self, cols=[]):
        Transformer.__init__(self)
        self.cols = cols
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
        cols = df.columns 
        if len(self.cols) > 0:
            cols = self.cols
        df = self.get_uniqueness(df.filter(items=cols)).to_frame().rename(columns={0: 'count'}).sort_values(by=['count'])
        self.display(df)
