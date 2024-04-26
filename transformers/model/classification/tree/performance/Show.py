import pandas as pd

from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.Transformer import Transformer

class Show(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        self.from_ds_init(ds)
        pdf = pd.concat([ds.perf_train, ds.perf_test], keys=['train', 'test'], ignore_index=False)
        self.display(pdf)
        ds.perf_df = pdf
        return ds        
    def from_ds_init(self, ds):
        self.ds = ds
        self._model = ds._model
