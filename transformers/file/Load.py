import pandas as pd
from deepspace.base import Base
from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class CSVLoader(Transformer):
    def __init__(self, path, sep=','):
        Base.__init__(self, sep='=', nb=50)
        Transformer.__init__(self)
        self.sep = sep
        self.path = path
    def transform(self, ds):
        if not ds:
            ds = DataSpace()    
        ds.data = self.extract()
        return ds        
    def extract(self):
        self.data = pd.read_csv(self.path, sep=self.sep)
        return self.data