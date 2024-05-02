import matplotlib.pyplot as plt
import seaborn as sns
from deepspace.DataSpace import DataSpace
from deepspace.transformers.column.abstract import Abstract

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
        if len(self.num_cols):
          corr = corr.filter(items=self.num_cols)
        plt.figure(figsize=(15, 7))
        sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap=self.cmap)
        plt.show()