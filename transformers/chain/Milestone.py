import pandas as pd
from deepspace.base import Base
from deepspace.transformers.Transformer import Transformer

class Milestone(Transformer):
    def __init__(self, ds=None, chainlink=None, clone=False):
        Base.__init__(self, sep='=', nb=50)
        Transformer.__init__(self)
        self.doclone = clone
        transformers = []
        if chainlink:
            ds = chainlink.ds
            transformers = chainlink.transformers
        if clone:
            ds = ds.clone()
        self.ds = ds
        self.transformers = transformers
    def clone(self):    
        return Milestone(ds=self.ds, clone=True)   
