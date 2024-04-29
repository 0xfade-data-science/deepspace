
from deepspace.transformers.drop.DropCols import DropCols
from deepspace.transformers.overview.CheckUniqueness import CheckUniqueness
from deepspace.DataSpace import DataSpace

class Drop(DropCols, CheckUniqueness):
    ''' a categorical variable is defined as such if count is less than defined threshold
        TODO add filter when card = 1
    '''
    def __init__(self, cat_cols=[], is_cat_threshold=(1,21), inplace=True):
        CheckUniqueness.__init__(self)
        DropCols.__init__(self, cat_cols, inplace=inplace)
        self.org_cat_cols = cat_cols
        self.cat_cols = cat_cols
        self.is_cat_threshold = is_cat_threshold #if count() < is_cat_threshold then remove
    def transform(self, ds: DataSpace):
        self.separator()
        self.ds = ds
        #pdb.set_trace()
        cols = self.get_drop_categ(ds.data)
        self.ds.data = self.drop(ds.data, cols)
        ds.add_drop_cols(cols)
        return self.ds
    def get_drop_categ(self, df):
        self.view_uniqueness(df)
        uniq = self.get_uniqueness(df)
        uniq = pd.DataFrame({'Features': uniq.index, 'Count': uniq.values})
        import pdb
        debug = False
        cols = []
        for row in uniq.iterrows():
            if debug:
                pdb.set_trace()
            rowid = row[0]
            col = row[1][0]
            cnt = row[1][1]
            if col not in self.get_cat_cols() :#only check categories
                continue
            if cnt <= self.is_cat_threshold[0] or cnt >= self.is_cat_threshold[1]  :
                self.print(f"Removing {col}")
                cols.append(col)
        return cols
    def drop(self, df, cols):
        if len(cols) <= 0:
            return df
        if self.inplace:
            df.drop(columns=cols, inplace=self.inplace)
        else:
            df = df.drop(columns=cols, inplace=self.inplace)
        self.print(df.columns.values.tolist())
        return df
    def get_cat_cols(self):
        if len(self.cat_cols) <= 0:
            if len(self.ds.cat_cols) <=0:
                self.cat_cols = self.org_cat_cols = self.df.select_dtypes(
                    include=["object", "category"]).columns.tolist()
            else:
                self.cat_cols = self.ds.cat_cols
        return self.cat_cols
