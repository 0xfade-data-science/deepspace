import pandas as pd
from DeepSpace.base import Base
from DeepSpace.transformers.Transformer import Transformer

class Milestone(Transformer):
    def __init__(self, ds=None, chainlink=None, clone=False):
        Base.__init__(self, sep='=', nb=50)
        Transformer.__init__(self)
        self.doclone = clone
        if chainlink:
            ds = chainlink.ds
        if clone:
            ds = ds.clone()
        self.ds = ds
    def transform(self, ds):    
        self.ds = ds
        return ds   