from sklearn.model_selection import train_test_split

from deepspace.DataSpace import DataSpace
from deepspace.transformers.Transformer import Transformer
from deepspace.Initialize import Initialize

class Splitter(Transformer):
    def __init__(self, test_size=0.2, seed=Initialize.seed, dostratification=False):
        Transformer.__init__(self)
        self.seed = seed
        self.test_size = test_size
        self.dostratification = dostratification
    def transform(self, ds: DataSpace):
        self.x, self.y = ds.x, ds.y
        self.split()
        ds.x_train, ds.y_train = self.x_train, self.y_train
        ds.x_test, ds.y_test = self.x_test, self.y_test
        return ds
    def split(self):
        self.separator()
        # Splitting the dataset into train and test datasets
        if self.dostratification:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=self.test_size, shuffle=True, random_state=self.seed, stratify=self.y)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=self.test_size, shuffle=True, random_state=self.seed)
