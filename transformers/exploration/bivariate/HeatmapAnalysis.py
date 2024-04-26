import matplotlib.pyplot as plt
import seaborn as sns
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.column.abstract import Abstract

class HeatmapAnalysis(Abstract):
    def __init__(self, num_cols=[], sep='=', nb=50):
        super().__init__(sep=sep, nb=nb)
        self.num_cols = num_cols
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.heatmap()
        return ds
    def heatmap(self):
        plt.figure(figsize=(15, 7))
        data = self.ds.data
        if len(self.num_cols) > 0:
            data = self.ds.data[self.num_cols]

        corr = data.corr(numeric_only=True)
        if len(self.num_cols):
          corr = corr.filter(items=self.num_cols)
        sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
        plt.show()