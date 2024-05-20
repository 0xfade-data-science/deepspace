from deepspace.transformers.exploration.bivariate.BivariateAnalysis import BivariateAnalysis

class NxC(BivariateAnalysis):
    def __init__(self, num_cols=[], cat_cols=[], ord_cols=[], only=[], figsize=(10,5), violin=True):
        BivariateAnalysis.__init__(self, num_cols=num_cols, cat_cols=cat_cols, ord_cols=ord_cols, 
                                    donvn=False, docvn=True, docvc=False, 
                                    doheatmap=False, only=only,
                                    figsize=figsize, violin=violin)

