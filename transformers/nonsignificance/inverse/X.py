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
        self.reverse()
        ds.isRnsfUnApplyed_XTest = True
        return self.ds
    def sanity_check(self, ds:DataSpace):
        pass
    def reverse(self):
        self.separator()
        self.ds.inv_test_data = pd.concat([self.ds.inv_test_data, self.ds.rnsf_x_test_removed], axis=1)
