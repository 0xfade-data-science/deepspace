import itertools  # cartesian product
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.exploration.bivariate.HeatmapAnalysis import HeatmapAnalysis

class BivariateAnalysis(HeatmapAnalysis):
    def __init__(self, num_cols=[], cat_cols=[], ord_cols=[], donvn=True, docvn=True, docvc=True, doheatmap=True, violin=True, only=[]):
        super().__init__('=', 50)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.ord_cols = ord_cols
        self.doheatmap=doheatmap
        self.donvn=donvn
        self.docvn=docvn
        self.docvc=docvc
        self.violin = violin
        self.only=only

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

    def analyse_num_vs_num(self):
        #num vs num
        pairs = self._get_num_pairs()
        self.print(pairs)
        for c1, c2 in pairs:
            #self.separator(n=1, string=f'num col "{c1}" vs num col "{c2}"')
            if c1 != c2:
                plt.figure(figsize=(7, 7))
                plt.scatter(self.ds.data[c1], self.ds.data[c2],  color='red')
                plt.xlabel(c1)
                plt.ylabel(c2)
                #plt.plot(Newspaper, newspaper_model.predict(Newspaper), color='blue', linewidth=3)
                plt.show()

    def analyse_cat_vs_num(self):
        target_col = self.ds.target_col
        #categ vs target col when numeric
        if target_col is not None:
            #if is_numeric_dtype(self.ds.data[target_col]):
            if target_col in self.num_cols:
                for col in self.cat_cols:
                    self.separator(n=1, string=f'cat "{col}" vs target "{target_col}"')
                    if col != target_col:
                        self.plot(col, target_col)
                        if self.violin:
                            self.show_violin(self.ds.data, x=target_col, y=col)

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
                        self.separator(n=1, sep='-', string=f'cat col "{col}" vs num col "{num_col}"')
                        self.plot(col, num_col)
                        if self.violin:
                            self.show_violin(self.ds.data, x=num_col, y=col)

    def analyse_cat_vs_cat(self):
        #categ vs categ
        pairs = self._get_cat_pairs()
        self.print(pairs)
        for c1, c2 in pairs:
            self.separator(n=1, sep='-', string=f'cat "{c1}" vs cat "{c2}"')
            if c1 != c2:
                self.stacked_barplot(c1, c2)
    def _get_pairs(self, cols):
        pairs = [(c1, c2) for i, c1 in enumerate(cols) for c2 in cols[i + 1:]]
        return pairs
    def _get_num_pairs(self):
        pairs = self._get_pairs(self.num_cols)
        return pairs
    def _get_cat_pairs(self):
        pairs = self._get_pairs(self.cat_cols)
        return pairs

    def heatmap(self):
        HeatmapAnalysis.heatmap(self)#, cols=self.num_cols)

    def plot(self, x, y):
        #pdb.set_trace()
        self.separator(n=1, string=f'plot for {x}/{y}')
        data = self.ds.data
        if x in self.ord_cols:
          sorter = data.sort_values(by=x, ascending=True).reset_index()
        else:
          sorter = data[[x, y]].groupby([x]).mean().sort_values(by=y, ascending=False).reset_index()
        # barplot shows only the mean
        sns.barplot(y=y, x=x, data=data, order=sorter[x])
        plt.xticks(rotation=90)
        plt.show()
    # Function to plot stacked bar plots

    def stacked_barplot(self, x, y):
        """
        Print the category counts and plot a stacked bar chart

        data: dataframe
        x: independent variable
        y: independent variable
        """
        self.separator(n=1, string=f'stacked barplot for {x}/{y}')
        data = self.ds.data
        X = data[x]
        Y = data[y]
        count = X.nunique()
        sorter = Y.value_counts().index[-1]
        tab1 = pd.crosstab(X, Y, margins=True).sort_values(by=sorter, ascending=False)
        self.display(tab1)
        print("_" * 120)
        tab = pd.crosstab(X, Y, normalize="index").sort_values(by=sorter, ascending=False)
        tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
        plt.legend(loc="lower left", frameon=False,)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.xticks(rotation=90)
        plt.show()

    def show_violin(self, df, x, y, figsize=(12, 7), log_scale=False):
        self.separator(n=1, string=f'violin plot for {x}/{y}')
        sns.violinplot(data=df, x=x, y=y, log_scale=log_scale)
        plt.show()


