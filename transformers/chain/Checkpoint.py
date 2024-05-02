import pandas as pd
from deepspace.base import Base
from deepspace.transformers.Transformer import Transformer   

class Checkpoint(Transformer):
    def __init__(self, transformer, clone=False):
        Base.__init__(self, sep='=', nb=50)
        Transformer.__init__(self)
        self.doclone = clone
        transformers = []
        ds = transformer.ds
        transformers = transformer.transformers
        if clone:
            ds = ds.clone()
        self.ds = ds
        self.transformers = transformers
    def clone(self):    
        return Checkpoint(self, clone=True)   