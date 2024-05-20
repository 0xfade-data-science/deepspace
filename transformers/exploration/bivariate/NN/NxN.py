from deepspace.transformers.exploration.bivariate.BivariateAnalysis import BivariateAnalysis

class NxN(BivariateAnalysis):
    def __init__(self, num_cols=[], only=[], figsize=(10,5), violin=True, violin_bins=10):
        BivariateAnalysis.__init__(self, num_cols=num_cols, cat_cols=[], 
                                donvn=True, docvn=False, docvc=False, 
                                doheatmap=False, only=only,
                                figsize=figsize, violin=violin, violin_bins=violin_bins)
