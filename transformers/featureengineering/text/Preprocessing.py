import pandas as pd 
import nltk

#nltk.download('all')
nltk.download('stopwords')
nltk.download('wordnet')

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from deepspace.transformers.Transformer import Transformer
from deepspace.DataSpace import DataSpace
from deepspace.transformers.featureengineering.Abstract import FuncTransformer as AbstractFuncTransformer

#source https://www.datacamp.com/tutorial/text-classification-python
class Preprocessing(AbstractFuncTransformer):
    '''
    language ='french' | 'english' | ...
    '''
    def __init__(self, feature, new_feature, feature_lang, language_short, language_long):
        AbstractFuncTransformer.__init__(self, feature, new_feature)
        self.feature_lang = feature_lang
        self.language_short = language_short
        self.language_long = language_long
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
        lemmatizer = WordNetLemmatizer()
        data = data.query(f'{self.feature_lang} == "{self.language_short}"')
        text = list(data[self.feature])
        corpus = []
        for i in range(len(text)):
            r = re.sub('[^a-zA-Z]', ' ', text[i])
            r = r.lower()
            r = r.split()
            r = [word for word in r if word not in stopwords.words(self.language_long)]
            r = [lemmatizer.lemmatize(word) for word in r]
            r = ' '.join(r)
            corpus.append(r)
        data[self.new_feature] = corpus
        data.head()        
        self.df = data