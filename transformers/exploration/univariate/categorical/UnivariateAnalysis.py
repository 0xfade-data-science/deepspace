import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.column.abstract import Abstract

class UnivariateAnalysis(Abstract):
    def __init__(self, cat_cols = [], ord_cols = [], normalize=True, only=[], debug=False):
        super().__init__()
        self.normalize = normalize
        self.cat_cols = cat_cols
        self.ord_cols = ord_cols
        self.only = only
        self.debug = debug
    def transform(self, ds:DataSpace):
        cols = self._get_cat_cols(ds)
        self.show_count(ds.data, cols, normalize=self.normalize)
        self.show_countplot(ds.data, cols)
        return ds
    def show_count(self, df, cat_cols, normalize=True):
        self.separator()
        if len(cat_cols) <= 0:
            raise Exception('empty cat_cols')
        for column in cat_cols:
            if normalize is None :
              print(df[column].value_counts(normalize=False))
              print(df[column].value_counts(normalize=True))
            else  :
              print(df[column].value_counts(normalize=True))
            print("-" * 50)
    def show_countplot(self, df, cat_cols):
        for col in cat_cols:
            self._countplot_pct(df, col)
            print("-" * 50)
    def _countplot_pct(self, df, feature, figsize=(12, 7), cpt=True):
        """
        Boxplot and histogram combined

        data: dataframe
        feature: dataframe column
        figsize: size of figure (default (12,7))
        kde: whether to the show density curve (default False)
        bins: number of bins for histogram (default None)
        """
        total = len(df[feature])  # Length of the column
        f1, (ax) = plt.subplots(
            nrows=1,      # Number of rows of the subplot grid = 2
            #sharex=True,  # x-axis will be shared among all subplots
            #gridspec_kw={"height_ratios": (0.25, 0.75)},
            figsize=figsize,
        )
        #ax.set_xticklabels(df, rotation=90)
        plt.xticks(rotation=90) #, ha='right')
        if feature in self.ord_cols:
          #sorter = df.sort_values(by=feature, ascending=True).reset_index()
          #sorter = df[feature].values().sort_values(ascending=True).index
          n = df[feature].unique()
          n.sort()
          sorter = pd.Series(n)
        else:
          sorter = df[feature].value_counts().index
        ax = sns.countplot(data=df, x=feature, ax=ax, palette='Paired', order = sorter) #color="violet",
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)  # Percentage of each class of the category
            x = p.get_x() + p.get_width() / 2 - 0.05  # Width of the plot
            y = p.get_y() + p.get_height()  # Height of the plot

            ax.annotate(percentage, (x, y), size=12, ha='center')  # Annotate the percentage with center alignment
        plt.show()
