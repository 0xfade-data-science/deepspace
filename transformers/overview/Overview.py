
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.overview.Shape import Shape
from deepspace.transformers.overview.Info import Info
from deepspace.transformers.overview.HeadTail import HeadTail
from deepspace.transformers.overview.Describe import DescribeNumerical, DescribeCategorical

from deepspace.transformers.chain.Milestone import Milestone
from deepspace.DataSpace import DataSpace

class Overview(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):
            self.separator(caller=str(self))
            _ = Milestone(ds) >> Shape() >> HeadTail() >> Info() >> DescribeNumerical() >> DescribeCategorical()
            return _.ds
    
class OverviewOLD(Transformer):
    ''''''
    def __init__(self, _):
        Transformer.__init__(self)
        self._ = _
    def transform(self, ds: DataSpace):
        self.separator(caller=str(self))
        df = ds.data
        self.display(df.head())
        self.display(df.tail())
        df.info()
        self.print('\n')
        self.print(df.shape)
        return ds