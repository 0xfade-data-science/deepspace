from sklearn.preprocessing import LabelEncoder


from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.encode.X.BeforeSplit import EncoderBeforeSplit as BaseEncoder

#############################################################
## This transformer should be used to encode target values ##
#############################################################
class EncoderBeforeSplit(BaseEncoder):
    ''' Only for Y, not tested yet'''
    def __init__(self, cat_cols=[], drop_first=True):
        BaseEncoder.__init__(self, cat_cols=cat_cols, drop_first=drop_first)
    def transform(self, ds:DataSpace):
        self.separator(caller=str(self))
        self.ds = ds
        self.df = self.ds.data
        self.data = self.ds.data
        cols = self.get_cat_cols()
        self.print("cat cols ", cols)
        # Warning on LabelEncoder "This transformer should be used to encode target values, i.e. y, and not the input X."
        # see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        for c in cols:
            labelencoder = LabelEncoder()
            self.data[c] = labelencoder.fit_transform(self.data[c])
        self.ds.data = self.data
        return ds
