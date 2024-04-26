from DeepSpace.transformers.exploration.bivariate.BivariateAnalysis import BivariateAnalysis

class NxN(BivariateAnalysis):
    def __init__(self, num_cols=[], only=[]):
        BivariateAnalysis.__init__(self, num_cols=num_cols, cat_cols=[], donvn=True, docvn=False, docvc=False, doheatmap=False, only=only)
