import pdb
import pandas as pd
import pickle
import os
import re

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.base import Base

class Save(Transformer):
    def __init__(self, identifer, saveto, kind):
        Base.__init__(self, sep='*', nb=50)
        Transformer.__init__(self)
        self.saveto = saveto
        self.identifer = identifer
        self.kind = kind

    def save(self, df):    
        self.print(f"Trying to save to {self.saveto}...")
        obj = self.load()
        df['kind'] = self.kind
        df['iteration'] = self.identifer 
        obj.performance[self.identifer] = df
        with open(self.saveto, 'wb') as outp:
            self.print(f"Saving to {self.saveto}")
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    def load(self):
        self.print(f"Trying to load from {self.saveto}...")
        if not os.path.isfile(self.saveto):
            ds = DataSpace()
            ds.performance = {}
            return ds
        with open(self.saveto, 'rb') as inp:
            self.print(f"Loading from {self.saveto}")
            ds = pickle.load(inp)
        return ds

