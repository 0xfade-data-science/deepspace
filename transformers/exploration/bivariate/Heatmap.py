import matplotlib.pyplot as plt
import seaborn as sns
from deepspace.DataSpace import DataSpace
from deepspace.transformers.column.Abstract import Abstract

class Heatmap(Abstract):
    def __init__(self, num_cols=[], figsize=(15, 7), annot=True, fmt=".2f", cmap='coolwarm', vmin=0, vmax=None, sep='=', nb=50):
        super().__init__(sep=sep, nb=nb)
        self.num_cols = num_cols
        self.figsize = figsize
        self.annot = annot
        self.fmt = fmt
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap 
    def transform(self, ds:DataSpace):
        self.ds = ds
        self.plot(ds.data_tmp)
        return ds
    def plot(self, df):
        self.separator(caller=self)
        plt.figure(figsize=self.figsize)
        sns.heatmap(df, annot=self.annot, fmt=self.fmt, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        plt.show()
        return self