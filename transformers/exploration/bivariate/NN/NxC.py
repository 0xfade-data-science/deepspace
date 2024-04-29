from deepspace.transformers.exploration.bivariate.BivariateAnalysis import BivariateAnalysis

class NxC(BivariateAnalysis):
    def __init__(self, num_cols=[], cat_cols=[], ord_cols=[], only=[]):
        BivariateAnalysis.__init__(self, num_cols=num_cols, cat_cols=cat_cols, ord_cols=ord_cols, donvn=False, docvn=True, docvc=False, doheatmap=False, only=only)

