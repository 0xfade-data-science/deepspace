
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.outliers.Check import CheckOutliers
from deepspace.transformers.duplicates.CheckDuplicated import CheckDuplicated
from deepspace.transformers.null.Check import CheckNulls
from deepspace.transformers.overview.CheckUniqueness import CheckUniqueness

from deepspace.transformers.chain.Milestone import Milestone
from deepspace.DataSpace import DataSpace

class SanityCheck(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):
            self.separator(caller=str(self))
            _ = Milestone(ds) >> CheckNulls() >> CheckUniqueness() >> CheckDuplicated() >> CheckOutliers()
            return _.ds
    
