from DeepSpace.transformers.exploration.bivariate.BivariateAnalysis import BivariateAnalysis

class CxC(BivariateAnalysis):
    def __init__(self, cat_cols=[], ord_cols=[], only=[]):
        BivariateAnalysis.__init__(self, num_cols=[], cat_cols=cat_cols, ord_cols=ord_cols, donvn=False, docvn=False, docvc=True, doheatmap=False, only=only)
