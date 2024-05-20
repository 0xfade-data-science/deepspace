
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.chain.Milestone import Milestone
from deepspace.DataSpace import DataSpace

from .Calc import Calc as PerformanceCalculator
from .View import View as PerformanceViewer

class CalcNView(Transformer):
    ''''''
    def __init__(self, identifer=None, saveto=None, kind='natural'):
        Transformer.__init__(self)
        self.saveto = saveto
        self.identifer = identifer
        self.kind = kind
                
    def transform(self, ds: DataSpace):
        _ = (   
            Milestone(ds)
            >> PerformanceCalculator()
            >> PerformanceViewer(self.identifer, self.saveto, self.kind)
        )
        return _.ds


