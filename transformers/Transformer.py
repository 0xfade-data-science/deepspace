
from DeepSpace.base import Base
import pdb

class Transformer(Base):
    def __init__(self, ds=None, debug=False):
        Base.__init__(self, sep='=', nb=50)
        self.ds = ds
        self.debug = debug
        self.transformers = [self]
    #@printcall()
    def transform(self, ds):
        self.ds = ds
        self.result = ds
        return ds
    def t(self, ds):
        return self.transform(ds)
    def set_ds(self, ds):
        self.ds = ds
    def __rshift__(self, t2):
        #pdb.set_trace()
        ds = self.t(self.ds)
        t2.set_ds(ds)
        t2.transformers = self.transformers +[t2] 
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
    
