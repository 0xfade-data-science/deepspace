import itertools  # cartesian product
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.base import Base
from deepspace.DataSpace import DataSpace
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.exploration.bivariate.Heatmap import Heatmap
from deepspace.transformers.exploration.plot.ScatterPlot import ScatterPlot
from deepspace.transformers.exploration.plot.BarPlot import BarPlot
from deepspace.transformers.exploration.plot.ViolinPlot import ViolinPlot
from deepspace.transformers.exploration.plot.StackedBarPlot import StackedBarPlot
from deepspace.transformers.column.abstract import Abstract

class BivariateAnalysis(Abstract):
    def __init__(self, num_cols=[], cat_cols=[], ord_cols=[], exclude=[], only=[], 
                 donvn=True, docvn=True, docvc=True, doheatmap=True, violin=True, 
                 figsize=(12, 7), cmap='coolwarm'):
        Abstract.__init__(self, target_col=None, cat_cols=cat_cols, num_cols=num_cols, 
                          exclude=exclude, only=[])          
        Base.__init__(self, '=', 50)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.ord_cols = ord_cols
        self.doheatmap=doheatmap
        self.donvn=donvn
        self.docvn=docvn
        self.docvc=docvc
        self.violin = violin
        self.only=only
        self.figsize=figsize
        self.cmap = cmap

    def transform(self, ds:DataSpace):
        self.ds = ds
        self.num_cols = self._get_num_cols(ds)
        self.cat_cols = self._get_cat_cols(ds)
        self.analyse()
        return ds
    
    def analyse(self):
      if self.doheatmap:
        self.heatmap()
      if self.donvn:
        self.analyse_num_vs_num()
      if self.docvn:
        self.analyse_cat_vs_num()
      if self.docvc:
        self.analyse_cat_vs_cat()

    def heatmap(self):
        Heatmap(self.num_cols, self.cmap).transform(self.ds) #, cols=self.num_cols)

    def analyse_num_vs_num(self):
        #num vs num
        pairs = self._get_num_pairs()
        self.print(pairs)
        for c1, c2 in pairs:
            #self.separator(n=1, string=f'num col "{c1}" vs num col "{c2}"')
            if c1 != c2:
                x = c1
                y = c2
                ScatterPlot(x, y, figsize=self.figsize).transform(self.ds)
                ViolinPlot(x, y, figsize=self.figsize).transform(self.ds)
    def analyse_cat_vs_num(self):
        target_col = self.ds.target_col
        #categ vs target col when numeric
        if target_col is not None:
            #if is_numeric_dtype(self.ds.data[target_col]):
            if target_col in self.num_cols:
                for col in self.cat_cols:
                    self.separator(n=1, string=f'cat "{col}" vs target "{target_col}"')
                    if col != target_col:
                        x = target_col
                        y = col
                        BarPlot(col, target_col, figsize=self.figsize).transform(self.ds)
                        ViolinPlot(x=x, y=y, figsize=self.figsize).transform(self.ds)

            else:
                self.print(f'target {target_col} not numeric')
        else:
            self.print(f'target {target_col} not defined')

        #categ vs num col
        #all but target
        for col in self.cat_cols:
            for num_col in self.num_cols:
                if num_col != target_col:
                    if is_numeric_dtype(self.ds.data[num_col]):
                        x = num_col
                        y = col
                        self.separator(n=1, sep='-', string=f'cat col "{y}" vs num col "{x}"')
                        BarPlot(x, y, figsize=self.figsize).transform(self.ds)
                        ViolinPlot(x, y, figsize=self.figsize).transform(self.ds)

    def analyse_cat_vs_cat(self):
        #categ vs categ
        pairs = self._get_cat_pairs()
        self.print(pairs)
        for c1, c2 in pairs:
            self.separator(n=1, sep='-', string=f'cat "{c1}" vs cat "{c2}"')
            if c1 != c2:
                StackedBarPlot(c1, c2, figsize=self.figsize).transform(self.ds)

    def _get_pairs(self, cols):
        pairs = [(c1, c2) for i, c1 in enumerate(cols) for c2 in cols[i + 1:]]
        return pairs
    def _get_num_pairs(self):
        pairs = self._get_pairs(self.num_cols)
        return pairs
    def _get_cat_pairs(self):
        pairs = self._get_pairs(self.cat_cols)
        return pairs



