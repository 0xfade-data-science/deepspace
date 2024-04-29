import itertools  # cartesian product
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

from deepspace.base import Base
from deepspace.DataSpace import DataSpace
from deepspace.transformers.column.abstract import Abstract

class UnivariateAnalysis(Abstract):
    def __init__(self, num_cols=[], exclude=[], simple_histos=True, boxplot_histos=True, violin=True, bins=None, limit=None):
        Base.__init__(self, '=', 50)
        Abstract.__init__(self, num_cols=num_cols, exclude=exclude)
        self.num_cols = num_cols
        self.exclude = exclude
        self.simple_histos = simple_histos
        self.boxplot_histos = boxplot_histos
        self.violin = violin
        self.limit = limit
        self.bins = bins
    def transform(self, ds:DataSpace):
        self.separator(caller=str(self))
        num_cols = self._get_num_cols(ds)
        self.show_count(ds.data, num_cols)
        self.analyse(ds.data, num_cols)
        return ds
    def show_count(self, df, num_cols, normalize=True):
        self.separator()
        if len(num_cols) <= 0:
            raise Exception('empty num_cols')
        for column in num_cols:
            if normalize is None :
              print(df[column].value_counts(normalize=False))
              print(df[column].value_counts(normalize=True))
            else  :
              print(df[column].value_counts(normalize=True))
            print("-" * 50)

    def set_simple_histos(self, v):
        self.simple_histos = v

    def boxplot_histos(self, v):
        self.boxplot_histos = v

    def analyse(self, df, num_cols):
        self.separator()
        if self.simple_histos:
            self.show_histos(df, num_cols)
        if self.boxplot_histos:
            for feature in num_cols:
                self.separator(n=1, string=f'histobox for {feature}')
                self.histogram_boxplot(df, feature, bins=self.bins)
        if self.violin:
            for feature in num_cols:
                self.separator(n=1, string=f'violin for {feature}')
                self.show_violin(df, feature=feature)
    def show_histos(self, _df, num_cols):
      df = _df[num_cols]
      if self.limit and len(num_cols) == 1:
          col = num_cols[0]
          df = _df.query(f"{col} <= {self.limit}")
      df[num_cols].hist(figsize=(14, 14), bins=self.bins)
      plt.show()
    # Function to plot a boxplot and a histogram along the same scale

    def histogram_boxplot(self, df, feature, figsize=(12, 7), kde=True, bins=None):
        """
        Boxplot and histogram combined

        data: dataframe
        feature: dataframe column
        figsize: size of figure (default (12,7))
        kde: whether to the show density curve (default False)
        bins: number of bins for histogram (default None)
        """
        f2, (ax_box2, ax_hist2) = plt.subplots(
            nrows=2,      # Number of rows of the subplot grid = 2
            sharex=True,  # x-axis will be shared among all subplots
            gridspec_kw={"height_ratios": (0.25, 0.75)},
            figsize=figsize,
        )                   # Creating the 2 subplots
        sns.boxplot(data=df, x=feature, ax=ax_box2, showmeans=True, color="violet")# Boxplot will be created and a star will indicate the mean value of the column

        if bins :
            sns.histplot(data=df, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter")
        else :
            sns.histplot(data=df, x=feature, kde=kde, ax=ax_hist2)# For histogram

        ax_hist2.axvline(df[feature].mean(), color="green", linestyle="--")# Add mean to the histogram
        ax_hist2.axvline(df[feature].median(), color="black", linestyle="-") # Add median to the histogram
        plt.show()
    def show_violin(self, df, feature, figsize=(12, 7), log_scale=False):
        sns.violinplot(data=df, x=feature, log_scale=log_scale)
        plt.show()