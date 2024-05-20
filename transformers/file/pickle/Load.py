import pdb
import pandas as pd
from deepspace.base import Base
from deepspace.transformers.Transformer import Transformer
import pickle

class Load(Transformer):
    def __init__(self, path, versioned=True):
        Base.__init__(self, sep='*', nb=50)
        Transformer.__init__(self)
        self.path = path
        self.versioned= versioned
   # @printcall()
    def transform(self, _):
        ds = None    
        self.print(f"Trying to load from {self.path}...")
        with open(self.path, 'rb') as inp:
            self.print(f"Loading from {self.path}")
            ds = pickle.load(inp)
            #print(ds)  # 
        return ds