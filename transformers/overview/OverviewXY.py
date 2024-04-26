from DeepSpace.transformers.overview import Overview
from DeepSpace.DataSpace import DataSpace

class OverviewXY(Overview):
    ''''''
    def __init__(self):
        Overview.__init__(self)
    def transform(self, ds: DataSpace):
        x,y = ds.x, ds.y
        self.display(x.head())
        self.print(x.shape)
        self.display(y.head())
        return ds