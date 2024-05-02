
from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class HeadTail(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        self.separator(caller=str(self))
        df = ds.data
        self.display(df.head())
        self.display(df.tail())
        return ds