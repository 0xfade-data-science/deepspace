import matplotlib.pyplot as plt
import seaborn as sns
from deepspace.DataSpace import DataSpace
from deepspace.transformers.column.Abstract import Abstract

class CorrHeatmap(Abstract):
    def __init__(self, num_cols=[], sep='=', nb=50, cmap='coolwarm'):
        super().__init__(sep=sep, nb=nb)
        self.num_cols = num_cols
        self.cmap = cmap #"Spectral"
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot()
        return ds
    def plot(self):
        self.separator(caller=self)
        data = self.ds.data
        if len(self.num_cols) >= 2:
            data = self.ds.data[self.num_cols]
        corr = data.corr(numeric_only=True)
        cols = self.get_num_cols()
        if len(cols):
          corr = corr.filter(items=cols)
        plt.figure(figsize=(15, 7))
        sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap=self.cmap)
        plt.show()