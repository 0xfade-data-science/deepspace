from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class Summary(Transformer):
    def __init__(self) : #, perfchecker : MyPerformanceChecker):
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):        
        self.fitted = ds._model
        self.print(self.fitted.summary())
        return ds