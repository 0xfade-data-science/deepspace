

from deepspace.transformers.Transformer import Transformer

class Adjust(Transformer):
    def __init__(self, drop_cols=[]):
        super().__init__()
        self.drop_cols=drop_cols
    def transform(self, ds):
        self.separator(caller=self)
        ds.set_drop_cols(self.drop_cols)
        return ds
class ViewDrop(Transformer):
    def __init__(self, drop_cols=[]):
        super().__init__()
        self.drop_cols=drop_cols
    def transform(self, ds):
        self.separator(caller=self)
        self.print(ds.get_drop_cols())
        return ds
