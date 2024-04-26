from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace

class Describe(Transformer):
    ''''''
    def __init__(self, include=[], only=[]):
        Transformer.__init__(self)
        self.include = include
        self.only = only
    def transform(self, ds: DataSpace):
        self.separator(caller=str(self))
        
        df = ds.data
        if len(self.include) > 0 :
          df = ds.data.filter(items=self.include)
        if len(self.only) > 0 :
          df = ds.data.filter(items=self.only)
        self.display(df.describe().T)
        return ds
