import pandas as pd
from deepspace.base import Base
from deepspace.transformers.Transformer import Transformer

class Monad(Transformer):
    def __init__(self, ds=None, monad=None, clone=False):
        Base.__init__(self, sep='=', nb=50)
        Transformer.__init__(self)
        self.doclone = clone
        transformers = []
        if monad:
            ds = monad.ds
            transformers = monad.transformers
        if clone:
            ds = ds.clone()
        self.ds = ds
        self.transformers = transformers
    def clone(self):    
        return Monad(ds=self.ds, clone=True)   
