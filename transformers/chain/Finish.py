from DeepSpace.transformers.Transformer import Transformer

class Finish(Transformer):
    def __init__(self, ds=None):
        Transformer.__init__(self, ds=ds)
        self.ds = ds
    def transform(self, ds):
        return ds
    
class Intermed(Finish):#equivalent to a finish, just tp differentiate last step from last intermediary step
    pass