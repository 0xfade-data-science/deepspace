import pandas as pd
from deepspace.base import Base
from .Monad import Monad

class Milestone(Monad):
    def __init__(self, ds=None, monad=None, clone=False):
        Monad.__init__(self, ds=ds, monad=monad, clone=clone)
