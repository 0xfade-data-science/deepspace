# Train Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

from deepspace.DataSpace import DataSpace
from deepspace.transformers.column.Abstract import Abstract

class Vectorizer(Abstract):
    '''not tested yet'''
    def __init__(self, num_cols=[], exclude=[]):
        Abstract.__init__(self, num_cols=num_cols, exclude=exclude)
    def transform(self, ds:DataSpace):
        self.separator()
        self.ds = ds
        self.x_train, self.y_train = ds.x_train, ds.y_train 
        self.x_test, self.y_test = ds.x_test, ds.y_test 
        self.vectorize()
        ds.isVectorized = True
        self.ds.x_train_v = self.x_train_v 
        self.ds.vectorizer = self.raw_vectorizer
        return ds
    def vectorize(self): 
        #Creating an instance of the MinMaxScaler
        self.raw_vectorizer = CountVectorizer()
        self.x_train_v = self.raw_vectorizer.fit_transform(self.x_train)
