import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from DeepSpace.DataSpace import DataSpace
from DeepSpace.base import Base
from DeepSpace.transformers.Transformer import Transformer

class Printer(Transformer):
    def __init__(self):
        Base.__init__(self, '#!#', 50)
        Transformer.__init__(self)
    def transform(self, ds:DataSpace):
        self.separator(sep='%', caller=self)
        print(ds.target_col)    
        return ds
