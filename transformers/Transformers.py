from deepspace.transformers.Transformer import Transformer
from deepspace.base import Base

class Transformers(Base):
    def __init__(self, *transformers):
        Base.__init__(self)
        self.transformers = [ ]
        if len(transformers) > 1:
            #sanity check
            for t in transformers:
                if isinstance(transformers, Transformer):
                    self.transformers.append(t)
                else:
                    raise(f"unexpected transformer type {ref(t)}")
        elif len(transformers) == 1:
            if isinstance(transformers[0], list):
                self.transformers = transformers
            elif isinstance(transformers, Transformer):
                self.transformers = [ transformers ]
    def is_empty(self):
        if len(self.transformers) <= 0:
            return True
        return False
    def add(self, t):
        self.transformers.append(t)