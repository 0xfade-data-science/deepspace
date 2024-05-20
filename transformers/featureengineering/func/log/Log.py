import numpy as np

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.featureengineering.func.Forward import Forward
from deepspace.transformers.featureengineering.func.Back<ward import Backward

class Log(Forward):
    '''Target Feature Engineering'''
    def __init__(self, feature, new_feature):
        Forward.__init__(self, feature, new_feature, np.log)

class InvLog(Backward):
      '''Target Feature Engineering'''
      def __init__(self, feature, new_feature):
        Backward.__init__(self, feature, new_feature, lambda x: np.exp(x))