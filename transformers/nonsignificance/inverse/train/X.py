import pandas as pd

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class XInverseRnsf(Transformer):
    '''not tested yet'''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.sanity_check(ds)
        self.inverse_remove_non_sginificant_features()
        ds.isRnsfUnApplyed_XTrain = True
        return self.ds
    def sanity_check(self, ds:DataSpace):
        pass
    def inverse_remove_non_sginificant_features(self):
        self.separator()
        data = self.ds.x_train
        self.ds.inv_train_data = pd.concat([data, self.ds.rnsf_x_train_removed], axis=1)
