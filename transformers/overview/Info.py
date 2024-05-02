
from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class Info(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        self.separator(caller=str(self))
        df = ds.data
        df.info()
        return ds