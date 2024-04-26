

from DeepSpace.transformers.Transformer import Transformer

class AdjustCols(Transformer):
    def __init__(self, num_cols=[], cat_cols=[], drop_cols=[]):
        super().__init__()
        self.num_cols=num_cols
        self.cat_cols=cat_cols
        self.drop_cols=drop_cols
    def transform(self, ds):
        ds.set_num_cols(self.num_cols)
        ds.set_cat_cols(self.cat_cols)
        ds.setdrop_cols(self.drop_cols)
        return ds
