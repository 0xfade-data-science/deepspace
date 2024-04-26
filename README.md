# Deep Space

Deep Space is an umbrella library that runs on top of major data science libraries as [scikit-learn learn](https://scikit-learn.org/), [statsmodel](https://www.statsmodels.org/), [tensorflow](https://www.tensorflow.org/) to site only a few. 
It is the result of the 4 months course I took from **MIT Education Professional** labeled **[MIT Advanced Data Science Program](https://professional-education-gl.mit.edu/mit-applied-data-science-course)**.

In some aspects this Deep Space library resembles the Pipeline library, which I discoverd quite late during the program but Deep Space uses another angle to solve the underlying problem. 
You can see it as a **syntactic sugar** with a bunch of **reusable classes**.

## Objective
The purpose of the library is to avoid repetitive tasks, a.k.a DRY (Don't Repeat Yourself), object oritented, a.k.a. OOP, and Monadic, i.e. using the Monada paradigm borrowed from functional programming world (Haskel for instance) - Monads and Classes are quite smilar concepts, tend to solve the same issues in computer science, and here they are used altogether to solve our main goals.

## Example
Let's say you work on a data science project in which you are asked to predict the sentiment in the transportation industry. 
And for that purpose you are given some inputs as a csv file with rows and features.

### The main process 

The general strategy or steps is as follows (besides understanding the business needs in details):
1. import major libraries
2. load the file
3. overview the data
4. analyse the data
5. preprocess the data
6. build the model
7. validate the model

### The usual way:
It goes like this :
```
#Libraries for reading and manipulating data
import numpy as np
import pandas as pd
..
#Import data viz libraries
import matplotlib.pyplot as plt
import seaborn as sns
..
#Libraries for building linera regression model
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
..
#Import library for pre processing data
from sklearn.model_selection import train_test_split
..
data = pd.read_csv("Data.csv")
#Make a copy 
data = kart.copy()
..
data.head()
..
data.tail()
..
data.shape
..
data.info()
..
#check null values
data.isnull.sum()
#check duplicates
data.duplicated().sum()

#check unique values
cat_col = list(data.select_dtypes("object").columns)
for column in cat_col:
    print(data[column].value_counts())
    print("-" * 50)
..
#univarie analysis
##univariate numerical features analysis
data.hist()
..
##univariate categorical features analysis
sns.heatmap(..)
..
#data preprocessing
#check and treat outliers
..
#split the data
..
#scale the data
..
#data building
..
```

### The DeepSpace way:
#### Overview of data
Deep Space library tries to overcome the tedious and repetitive tasks and offers a pre-defined set of classes to make it easy and faster:
```
from DeepSpace.Initialize import Initialize
Initialize(seed=1)

from DeepSpace.transformers.file.Load import CSVLoader
from DeepSpace.transformers.chain.Milestone import Milestone
from DeepSpace.transformers.overview.Overview import Overview
from DeepSpace.transformers.overview.Describe import Describe
from DeepSpace.transformers.outliers.Check import CheckOutliers
from DeepSpace.transformers.duplicates.CheckDuplicated import CheckDuplicated
from DeepSpace.transformers.null.Check import CheckNulls
from DeepSpace.transformers.overview.CheckUniqueness import CheckUniqueness

file = "travel_train_data"
#overview of the data 
view_chain_result =  (
     >> CSVLoader(file, ",")
     >> Overview() 
     >> Describe() 
     >> CheckUniqueness()
     >> CheckNulls() 
     >> CheckDuplicated()
     >> CheckOutliers()
     >> Milestone()
)
```
The >> chevrons are the expression of the Monads strategy.

The above code can also be simplified by creating a class a not rewriting the whole code gain:
```
class SuperOverview(Trasnformer):
  def __init__(self, file, sep=','):
      super().__init__()
      self.file = file
      self.sep = sep
  def transform(self, ds:DataSpace):
      return (
         >> CSVLoader(self.file, self.sep)
         >> Overview() 
         >> Describe() 
         >> CheckUniqueness()
         >> CheckNulls() 
         >> CheckDuplicated()
         >> CheckOutliers()
         >> Milestone()
    )
```
Now you can use SuperOverview class like this in your future code :
```
    (SuperOverview(file) >> Milestone())
```

#### Data analysis, or Exploratory Data Analysis (EDA) 

Just like so to show a bunch of plots (histograms, boxplots, etc) and calculations for the numerical features:
```
    (milestone >> numerical.Univariate(num_cols=[]) >> Milestone())
```
You can set the num_cols parameter to only show specific featrues.

And the below code is for categorical features:
```
    (milestone >> categorical.Univariate() >> Milestone())
```
The same hereafter for bivariate analysis:
```
    (milestone >> Bivariate(num_cols=(), cat_cols=[]) >> Milestone())
```
### Data preprocessing
#### Drop non useful features
Drop ID feature: 
```
milestone = (
    milestone
    >> DropAdjuster(['ID'])
    >> DropCols()
    >> Milestone()
)
```
Drop ID and Price_OLD feature: 
```
milestone = (
    milestone
    >> DropAdjuster(['ID','Price_OLD'])
    >> DropCols()
    >> Milestone()
)
#### Null imputation and dupàlicates removal
Impute nulls (by default categorical with be imputed with the mode and numerical with the mean):
```
saveto = 'milestone-imputation-no-dupes.pkl'
milestone = (
    milestone
    >> ProcessImputation()
    >> DropDuplicates()
    >> Save(saveto)
    >> Milestone()
)
You can also save the result to a file (pckle file) to be reused later.
#### Feature engineering
milestone = (
    milestone
                >> Log('Departure_Delay_in_Mins', 'Departure_Delay_in_Mins_Log')                
                >> Log('Arrival_Delay_in_Mins', 'Arrival_Delay_in_Mins_Log')
                >> Log('Travel_Distance', 'Travel_Distance_Log')                
                >> Sum(['Arrival_Delay_in_Mins', 'Departure_Delay_in_Mins'], 'X_Delay_Sum')
                >> Log('X_Delay_Sum', 'X_Delay_Sum_Log', 10)                
                >> Polynom(['Departure_Delay_in_Mins_Log', 'Arrival_Delay_in_Mins_Log'], 'X_Delay_Ploy')
                >> Sine('Travel_Distance', 'Travel_Distance_Sine')                
                >> Milestone()
)
#### Separating and Splitting
milestone = (
    milestone
        >> EncoderBeforeSplit()
        >> DropAdjuster(drop_cols=[
                'Cleanliness_Extremely Poor',
                'CheckIn_Service_Extremely Poor',
                'Onboard_Service_Extremely Poor',
                'Online_Support_Extremely Poor',
                'Platform_Location_Very Inconvenient'
                        ])
        >> DropCols() 
        >> Scaler()
        >> TargetAdjuster('Overall_Experience_1')                
        >> Separator()
        >> Splitter() 
        >> Save('clean_train_chain_result-post-merge-split.pkl')
        >> Milestone()
)
#### Model Building and testing
from DeepSpace.transformers.model.classification.logistic.sklearn.LogisticRegression.Model import LogisticRegression
from DeepSpace.transformers.model.classification.logistic.sklearn.LogisticRegression.performance.Calc import Calc as PerfCalculator
#from DeepSpace.transformers.model.regression.linear.sklearn.LogisticRegression.performance.Show import Show as PerfViewer
from DeepSpace.transformers.model.classification.logistic.sklearn.LogisticRegression.performance.ShowCoeffs import ShowCoeffs, ShowInvCoeffs
#from DeepSpace.transformers.file.Save import Save, Load

milestone = (
    milestone
            Load('clean_train_chain_result-post-merge-split.pkl')
            >> LogisticRegression()
            #>> Debug()
            >> PerfCalculator()
            >> ShowCoeffs()
            >> ShowInvCoeffs()
            >> Finish()
)

