from DeepSpace.transformers.Transformer import Transformer

class Adjust(Transformer):
    def __init__(self, num_cols=[]):
        super().__init__()
        self.num_cols=num_cols
    def transform(self, ds):
        ds.set_num_cols(self.num_cols)
        return ds