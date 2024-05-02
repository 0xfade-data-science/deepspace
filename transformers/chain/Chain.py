import pandas as pd

from deepspace.transformers.Transformer import Transformer
from deepspace.transformers.model.Abstract import Abstract as AbstractModel
from deepspace.transformers.chain.Finish import Finish
from deepspace.base import Base

class Chain(Transformer):
    def __init__(self, name='root', transformers=None):
        Transformer.__init__(self)
        self.name = name
        self.transformers = transformers
        self.chains = []
    def add(self, chain):
       self.chains.append(chain)
    def show(self, show_transformers=True, level=0):
        shifts = level*' ' 
        print(shifts+self.name)
        if len(self.chains) <=0 and show_transformers:
           csv = ','.join([str(t) for t in self.transformers])
           print(shifts+csv)
        else:
            l = 1
            for chain in self.chains :
                chain.show(show_transformers=show_transformers, level=l)
                l=l+1
    def show_perfs(self):
        chains = self.chains
        m = None
        models = [ ]
        for chain in chains :
          transformers = chain.transformers
          ds = transformers[-1].ds
          for m in transformers :
            if isinstance(m, AbstractModel) :
              models.append((m, ds))

        for i,m in enumerate(models):
            (m1, ds1) = models[i]
            m1name = f'{m1.__class__.__name__} (#{i})'
            pdf1 = pd.concat([ds1.perf_train, ds1.perf_test], keys=['train', 'test'], ignore_index=False)

            self.display(pdf1)                