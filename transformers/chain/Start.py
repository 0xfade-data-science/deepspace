from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.file.File import File

class Start_OLD(Transformer):
    def __init__(self, ds=None):
        Transformer.__init__(self, ds=ds)
    def transform(self, file:File):
        return DataSpace().t(file)

class Start(Transformer):
    def __init__(self, ds=None):
        Transformer.__init__(self, ds=ds)
    def transform(self, _): #unused parameter
        return DataSpace()
    
class Continue(Transformer):
    def __init__(self, ds=None, monad=None):
        if ds:
            Transformer.__init__(self, ds=ds)
        elif monad:
            Transformer.__init__(self, ds=monad.ds)            
    def transform(self, _): #unused parameter
        return self.ds

