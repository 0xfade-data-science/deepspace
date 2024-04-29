from deepspace.transformers.Transformer import Transformer

class Adjust(Transformer):
    def __init__(self, cat_cols=[]):
        super().__init__()
        self.cat_cols=cat_cols
    def transform(self, ds):
        ds.set_cat_cols(self.cat_cols)
        return ds
