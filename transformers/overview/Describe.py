from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace

class Describe(Transformer):
    ''''''
    def __init__(self, include=[], include_types=None, exclude_types=None, only=[]):
        Transformer.__init__(self)
        self.include = include
        self.only = only
        self.include_types = include_types
        self.exclude_types = exclude_types
    def transform(self, ds: DataSpace):
        self.separator(caller=str(self))        
        df = self.get_df(ds)
        self.display(df.describe(include=self.include_types, exclude=self.exclude_types).T)
        return ds
    def get_df(self, ds: DataSpace):
        df = ds.data
        if len(self.include) > 0 :
          df = ds.data.filter(items=self.include)
        if len(self.only) > 0 :
          df = ds.data.filter(items=self.only)
        return df

class DescribeNumerical(Describe):
    ''''''
    def __init__(self, include=[], only=[]):
        Describe.__init__(self, include=include, include_types='number', only=only)
class DescribeCategorical(Describe):
    ''''''
    def __init__(self, include=[], only=[]):
        Describe.__init__(self, include=include, exclude_types='number', only=only)
