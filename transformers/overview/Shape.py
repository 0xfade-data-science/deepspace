
from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class Shape(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        self.separator(caller=str(self))
        df = ds.data
        self.print(df.shape)
        return ds