from deepspace.DataSpace import DataSpace
from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.chain.Wrap import Wrap
from deepspace.transformers.file.pickle.Load import Load

class Performance(Transformer):
    ''''''
    def __init__(self, storage=None, kinds=None):
        Transformer.__init__(self)
        self.storage = storage
        self.kinds = kinds
    def transform(self, ds: DataSpace):
        self.view()
        return ds
    def view(self):
        _ = (
            Load(self.storage)
            >> Wrap()
        )

        for key in _.ds.performance:
            df = _.ds.performance[key]
            if self.kinds:
                df = df.query('kind in @self.kinds')
            if df.index.size > 0:
                self.display(df)

