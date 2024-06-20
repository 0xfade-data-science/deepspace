import pandas as pd 
from langdetect import detect_langs

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.featureengineering.func.Abstract import FuncTransformer
from deepspace.transformers.featureengineering.Abstract import FuncTransformer as AbstractFuncTransformer


class LanguageDetector(AbstractFuncTransformer):
    def __init__(self, feature, new_feature, new_feature_pct):
        super().__init__(feature, new_feature)
        self.new_feature_pct = new_feature_pct
    def transform(self, ds:DataSpace):
        self.init_from_ds(ds)
        self.apply()
        self.ds_init()
        return self.ds

    def init_from_ds(self, ds):
        self.ds = ds
        self.df = self.ds.data 
    def ds_init(self):
        self.ds.data = self.df 

    def apply(self):
        self.separator(caller=self, string=f'applying function : {self.new_feature} = {self.func}({self.feature})')
        self.func(self.df)
        return self.df

    def func(self, data):
        def calc_lang(row):
            txt = row[self.feature]
            lang, pct = self.detect_language_with_langdetect(txt)
            return lang
        def calc_lang_pct(row):
            txt = row[self.feature]
            lang, pct = self.detect_language_with_langdetect(txt)
            return pct

        data[self.new_feature] = data.apply(calc_lang, axis=1)
        data[self.new_feature_pct] = data.apply(calc_lang_pct, axis=1)
 
    def detect_language_with_langdetect(self, line): 
        try: 
            langs = detect_langs(line) 
            for item in langs: 
                # The first one returned is usually the one that has the highest probability
                return item.lang, item.prob 
        except: return "err", 0.0 