

from DeepSpace.transformers.Transformer import Transformer

class Adjust(Transformer):
    def __init__(self, drop_cols=[]):
        super().__init__()
        self.drop_cols=drop_cols
    def transform(self, ds):
        ds.set_drop_cols(self.drop_cols)
        return ds
