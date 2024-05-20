import matplotlib.pyplot as plt
import seaborn as sns
from deepspace.DataSpace import DataSpace
from deepspace.transformers.column.Abstract import Abstract
from deepspace.transformers.chain.Monad import Monad

class Pivot(Abstract):
    def __init__(self, index, columns, values, filter=None):
        Abstract.__init__(self)
        self.index, self.columns, self.values = index, columns, values
        self.filter = filter
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot(self.index, self.columns, self.values)
        ds.data_tmp = self.dfpv
        return ds
    def plot(self, index, columns, values):
        self.separator(caller=self)
        df = self.ds.data.query(self.filter)
        self.dfpv = df.pivot_table(index=index, columns=columns, values=values)
        self.display(self.dfpv)
        return self