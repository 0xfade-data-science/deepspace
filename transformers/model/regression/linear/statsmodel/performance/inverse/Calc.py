import pandas as pd

from deepspace.DataSpace import DataSpace

from deepspace.transformers.model.regression.linear.statsmodel.performance.Save import Save 
from deepspace.transformers.model.regression.linear.statsmodel.performance.Calc import Calc as PerformanceCalculator
from deepspace.transformers.model.regression.linear.statsmodel.performance.inverse.train.Calc import Calc as InversePerfCalc_Train
from deepspace.transformers.model.regression.linear.statsmodel.performance.inverse.test.Calc import Calc as InversePerfCalc_Test

class Calc(PerformanceCalculator):
    def __init__(self, target_col, doview=True, identifier=None, saveto=None, kind='naturalized'):
        PerformanceCalculator.__init__(self)
        self.target_col = target_col
        self.doview = doview
        self.saveto = saveto
        self.identifier = identifier
        self.kind = kind

    def transform(self, ds:DataSpace):
        self.from_ds_init(ds)
        ds.inv_perf_train = self.perf_train()
        ds.inv_perf_test = self.perf_test()
        self.inv_perf = pd.concat([ds.inv_perf_train, ds.inv_perf_test], axis=0)
        ds.inv_perf = self.inv_perf
        self.show()
        self.save()
        return ds
    def perf_train(self):
        c = InversePerfCalc_Train(self.target_col)
        c.ds = self.ds
        return c.calc_perf()
    def perf_test(self):
        c = InversePerfCalc_Test(self.target_col)
        c.ds = self.ds
        return c.calc_perf()
    def show(self):
        if self.doview:
            self.display(self.inv_perf)
    def save(self):
        if self.saveto:
            Save(self.identifier, self.saveto, self.kind).save(self.inv_perf)
