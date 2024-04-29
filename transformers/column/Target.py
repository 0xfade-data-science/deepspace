from deepspace.transformers.Transformer import Transformer

class Adjust(Transformer):
    def __init__(self, target_col=None):
        super().__init__()
        self.target_col=target_col
    def transform(self, ds):
        ds.set_target_col(self.target_col)
        return ds
