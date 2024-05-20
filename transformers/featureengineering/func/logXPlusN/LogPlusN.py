import numpy as np

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.featureengineering.func.Forward import Forward
from deepspace.transformers.featureengineering.func.Back<ward import Backward

class LogPlusN(Forward):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature, N=1):
        Forward.__init__(self, feature, new_feature, lambda x: np.log(N+x))

class InvLogPlusN(Backward):
      '''Target Feature Engineering'''
      def __init__(self, feature, new_feature, N=1):
        Backward.__init__(self, feature, new_feature, lambda x: np.exp(x)-N)