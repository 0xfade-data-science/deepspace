import pandas as pd
from DeepSpace.base import Base
from DeepSpace.transformers.Transformer import Transformer

class CSVLoader(Transformer):
    def __init__(self, path, sep=','):
        Base.__init__(self, sep='=', nb=50)
        Transformer.__init__(self)
        self.sep = sep
        self.path = path
    def transform(self, ds):    
        ds.data = self.extract()
        return ds        
    def extract(self):
        self.data = pd.read_csv(self.path, sep=self.sep)
        return self.data