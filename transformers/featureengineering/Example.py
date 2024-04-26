from DeepSpace.transformers.Transformer import Transformer

#Specific to current projet
class FeatureEngineerBrandFromName(Transformer):
    '''Target Feature Engineering'''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds):
        df = ds.data
        dfm = df['Name'].str.split(pat = ' ', expand = True)
        df['Brand'] = dfm[0].str.upper()
        #df['Model'] = dfm[1].str.upper()
        return ds

#Specific to current projet
import datetime as dt
class FeatureEngineerAgeFromYear(Transformer):
    '''Target Feature Engineering'''
    def __init__(self):
        Transformer.__init__(self)
    def transform(self, ds):
        df = ds.data
        maxY = str(df['Year'].max())
        #print(maxY)
        dfDDMMYYYY = '01'+'-01-'+df['Year'].astype(str)
        df['Age']  = dt.datetime.strptime(maxY, "%Y")-pd.to_datetime(dfDDMMYYYY, dayfirst=True)
        df['Age'] = df['Age'].dt.days
        return ds