from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

from DeepSpace.transformers.Transformer import Transformer
from DeepSpace.base import Base
from DeepSpace.DataSpace import DataSpace
from DeepSpace.transformers.model.regression.tree.Show import Show


class SaveTree(Show):
    def __init__(self, saveto, max_depth=3, figsize=(25, 20)) :
        Base.__init__(self, '=', 50)
        Show.__init__(self, saveto=saveto, max_depth=max_depth, figsize=figsize)


