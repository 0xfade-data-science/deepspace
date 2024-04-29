import pdb

from deepspace.base import Base
from deepspace.transformers.Transformer import Transformer

class Debug(Transformer):
    def __init__(self):
        Base.__init__(self, sep='=', nb=50)
        Transformer.__init__(self)
   # @printcall()
    def transform(self, ds):
        pdb.set_trace()
        return ds