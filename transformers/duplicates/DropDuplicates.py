from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace

class DropDuplicates(Transformer):
    def __init__(self, keep='first', inplace=True, ignore_index=True):
        Transformer.__init__(self)
        self.keep=keep
        self.inplace=inplace
        self.ignore_index=ignore_index
    def transform(self, ds: DataSpace):
        df = ds.data
        if self.inplace:
            df.drop_duplicates(keep=self.keep, inplace=self.inplace, ignore_index=self.ignore_index)
        else:
            df = df.drop_duplicates(keep=self.keep, inplace=self.inplace, ignore_index=self.ignore_index)
        return ds