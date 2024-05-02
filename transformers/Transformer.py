
from deepspace.base import Base
#import pdb

class Transformer(Base):
    def __init__(self, ds=None, debug=False):
        Base.__init__(self, sep='=', nb=50)
        self.ds = ds
        self.debug = debug
        self.transformers = [self]
        self.transformed = False
    def transform(self, ds):
        self.ds = ds
        self.result = ds
        return ds
    def t(self, ds):
        return self.transform(ds)
    def set_ds(self, ds):
        self.ds = ds
    def get_ds(self):
        return self.ds 
    def get_data(self):
        return self.get_ds().get_data()
    def clone(self):
        return Transformer(ds=self.ds)
    def __rshift__(self, t2):
        ds = self.ds
        #execute both transformers
        if not self.transformed :
            ds = self.t(self.ds)
            self.transformed = True
        if not t2.transformed :
            t2.set_ds(ds)
            ds = t2.t(ds)
            t2.transformed = True
            t2.transformers = list(set(self.transformers +[t2])) 
        return t2
    def get_transformers(self):
        return self.transformers
    
class TransformerBeforeSplit(Transformer):
    def __init__(self):
        Base.__init__(self, sep='=', nb=50)
    #@printcall()
    def transform(self, ds):
        self.ds = ds
        self.result = ds
        return ds
    def from_ds_init(self, ds):
        self.ds = ds
        self.data = ds.data
    def init_ds(self, ds):
        pass

class TransformerAfterSplit(Transformer):
    def __init__(self):
        Transformer.__init__()
    def transform(self, ds):
        self.ds = ds
        self.result = ds
        return ds
    def from_ds_init(self, ds):
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self.ds = ds
    def init_ds(self, ds):
        ds.x_train, ds.y_train, ds.x_test, ds.y_test = self.x_train, self.y_train, self.x_test, self.y_test
    
