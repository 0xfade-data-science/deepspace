from DeepSpace.transformers.Transformer import Transformer

class Meta(Transformer):
    def __init__(self, transformers: list=[]):
        Transformer.__init__(self)
        self.transformers = transformers
    def is_empty(self):
        if len(self.transformers) <= 0:
            return True
        return False
    def transform(self, obj):
        for t in self.transformers:
            self.print(t)
            obj = t.transform(obj)
        self.result = obj
        return obj
    def add(self, t):
        self.transformers.append(t)