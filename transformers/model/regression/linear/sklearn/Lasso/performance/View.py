import pandas as pd

from deepspace.DataSpace import DataSpace

from deepspace.transformers.model.regression.linear.statsmodel.performance.Save import Save

class View(Save):
    ''''''
    def __init__(self, identifer=None, saveto=None, kind='natural'):
        Save.__init__(self, identifer, saveto, kind)

    def transform(self, ds: DataSpace):
        pdf = pd.concat([ds.perf_train, ds.perf_test], keys=['train', 'test'], ignore_index=False)
        self.display(pdf)
        ds.lasso_perf_df = pdf
        if self.saveto:
            self.save(pdf)
        return ds
