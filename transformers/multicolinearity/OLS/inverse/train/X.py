import pandas as pd

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class XInverseRmc(Transformer):
    '''not tested yet'''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.sanity_check(ds)
        self.inverse_remove_multicolinearity()
        ds.isRmcUnApplyed_XTrain = True
        return self.ds
    def sanity_check(self, ds:DataSpace):
        pass
    def inverse_remove_multicolinearity(self):
        self.separator()
        data = self.ds.x_train
        if self.ds.isRnsfUnApplyed_XTrain:
            data = self.ds.inv_train_data

        self.ds.inv_train_data = pd.concat([data, self.ds.rmc_x_train_removed], axis=1)
