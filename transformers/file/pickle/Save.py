import pdb
import pandas as pd
from deepspace.base import Base
from deepspace.transformers.Transformer import Transformer
import pickle
import os
import re

class Save(Transformer):
    def __init__(self, path):
        Base.__init__(self, sep='*', nb=50)
        Transformer.__init__(self)
        self.path = path
   # @printcall()
    def transform(self, ds):    
        obj = ds
        self.print(f"Trying to save to {self.path}...")
        with open(self.path, 'wb') as outp:
            self.print(f"Saving to {self.path}")
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        return ds

class SaveVersionedTODO(Transformer):
    def __init__(self, path, prefix):
        Base.__init__(self, sep='*', nb=50)
        Transformer.__init__(self)
        self.path = path
        version='00001'
        filename = f'{prefix}-v{version}.pkl'
   # @printcall()
    def transform(self, ds):    
        obj = ds
        self.print(f"Trying to save to {self.path}...")
        with open(self.path, 'wb') as outp:
            self.print(f"Saving to {self.path}")
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        return ds
    def find_version(self):
        rootdir = self.path
        regex = re.compile('(.*-v\d+\.pkl$)')
        files = []
        for root, dirs, files in os.walk(rootdir):
            for file in files:
                if regex.match(file):
                    files.append(file)

class Load(Transformer):
    def __init__(self, path, versioned=True):
        Base.__init__(self, sep='*', nb=50)
        Transformer.__init__(self)
        self.path = path
        self.versioned= versioned
   # @printcall()
    def transform(self, _):
        ds = None    
        self.print(f"Trying to load from {self.path}...")
        with open(self.path, 'rb') as inp:
            self.print(f"Loading from {self.path}")
            ds = pickle.load(inp)
            print(ds)  # 
        return ds