from deepspace.transformers.Transformer import Transformer  
#from deepspace.DataSpace import DataSpace provoques recursive import
from deepspace.base import Base

class Abstract(Transformer):
    def __init__(self, target_col=None, cat_cols=[], num_cols=[], exclude=[], only=[], sep='=', nb=50):
        Base.__init__(self, sep=sep, nb=nb)
        Transformer.__init__(self)
        if not type(num_cols) == list:
            raise Exception('num_cols should be a list of features')
        if not type(cat_cols) == list:
            raise Exception('cat_cols should be a list of features')
        self.target_col = target_col
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.exclude = exclude
        self.only = only

    def _get_target_col(self, ds):
        if self.target_col:
            return self.target_col
        return ds.target_col
    def _get_cols(self, mycols, refcols, df, dtypes):
        if len(mycols) <= 0:
            if len(refcols) <=0:
                mycols = df.select_dtypes(include=dtypes).columns.tolist()
            else:
                mycols = refcols
        cols = []
        for col in mycols:
            if col in df.columns :
                    if len(self.only)>0 and col not in self.only:
                      print(f'only cols excludes col {col}')
                      continue
                    if len(self.exclude)>0 and col in self.exclude:
                      print(f'excluded col {col}')
                      continue
                    cols.append(col)
        mycols = cols
        return mycols
    def _get_cat_cols(self, ds):
        return self._get_cols(self.cat_cols, ds.cat_cols, ds.data, ["object", "category"])
    def _get_num_cols(self, ds):
        return self._get_cols(self.num_cols, ds.num_cols, ds.data, ["number"])
