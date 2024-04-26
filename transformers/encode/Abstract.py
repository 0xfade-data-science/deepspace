from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.DataSpace import DataSpace

class AbstractEncoder(Transformer):
    def __init__(self, cat_cols=[]):
        Transformer.__init__(self)
        #if not cat_cols:
        #    raise Exception("cat_cols null")
        self.cat_cols = cat_cols
    def get_cat_cols(self):
        if len(self.cat_cols) <= 0:
            if len(self.ds.cat_cols) <=0:
                self.cat_cols = self.org_cat_cols = self.df.select_dtypes(
                    include=["object", "category"]).columns.tolist()
            else:
                self.cat_cols = self.ds.cat_cols
        cols = []
        for col in self.cat_cols:
            if col in self.df.columns:
                    cols.append(col)
        self.cat_cols = cols
        return self.cat_cols
    def transform(self, ds):
        return ds
