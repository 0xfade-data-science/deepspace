from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.file.File import File

class Start(Transformer):
    def __init__(self, ds=None):
        Transformer.__init__(self, ds=ds)
    def transform(self, file:File):
        return DataSpace().t(file)

class Start2(Transformer):
    def __init__(self, ds=None):
        Transformer.__init__(self, ds=ds)
    def transform(self, _): #unused parameter
        return DataSpace()
