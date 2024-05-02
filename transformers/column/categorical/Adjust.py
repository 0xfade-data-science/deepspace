import pandas as pd
from deepspace.transformers.Transformer import Transformer

class Adjust(Transformer):
    def __init__(self, cat_cols=[]):
        super().__init__()
        self.cat_cols=cat_cols
    def transform(self, ds):
        ds.set_cat_cols(self.cat_cols)
        return ds

class AdjustCategorical(Adjust):
    def __init__(self, cat_cols=[], ordered={}):
        super().__init__(cat_cols=cat_cols)
        self.cat_cols=cat_cols
        self.ordered = ordered
    def transform(self, ds):
        Adjust.transform(self, ds) 
        for cat in self.cat_cols:
            cats = ds.data[cat].unique().tolist()
            ordered = False
            if cat in self.ordered:
                ordered = True
                cats = self.ordered[cat]
            ds.data[cat] = pd.Categorical(ds.data[cat], 
                                            ordered = ordered, 
                                            categories = cats)

        return ds
