from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class CheckDuplicated(Transformer):
    ''''''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds: DataSpace):
        self.check_dupes(ds.data)
        return ds
    def check_dupes(self, df):
        self.separator(caller=str(self))
        v = df.duplicated().sum()
        self.print(f"Duplicates found : {v}")
