
from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace

class Overview(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        self.separator(caller=str(self))
        df = ds.data
        self.display(df.head())
        self.display(df.tail())
        df.info()
        self.print('\n')
        self.print(df.shape)
        return ds