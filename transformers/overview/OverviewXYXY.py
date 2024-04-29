from deepspace.transformers.overview.Overview import Overview
from deepspace.DataSpace import DataSpace

class OverviewXYXY(Overview):
    ''''''
    def __init__(self):
        Overview.__init__(self)
    def transform(self, ds:DataSpace):
        self.separator(caller=self)
        self.x_train, self.y_train, self.x_test, self.y_test = ds.x_train, ds.y_train, ds.x_test, ds.y_test
        self.display(self.x_train.head())
        self.print(self.x_train.shape)
        self.display(self.y_train.head())

        self.display(self.x_test.head())
        self.print(self.x_test.shape)
        self.display(self.y_test.head())

        return ds
