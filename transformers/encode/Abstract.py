from deepspace.transformers.column.Abstract import Abstract
from deepspace.DataSpace import DataSpace

class AbstractEncoder(Abstract):
    def __init__(self, cat_cols=[]):
        Abstract.__init__(self, cat_cols=cat_cols)
        #if not cat_cols:
        #    raise Exception("cat_cols null")
        self.cat_cols = cat_cols
    def transform(self, ds):
        return ds
