# Deep Space

Deep Space is an umbrella library that runs on top of major data science libraries as <a href="https://scikit-learn.org/"> scikit-learn learn </a> and <a href="https://www.statsmodels.org/">statsmodel<a> to site only few. 
It is the result of the course I took from MIT Education Professional labeled MIT Advanced Data Science Program.

In some aspects the Deep Space resemble the Pipeline library, which I discoverd quite late during the program but Deep Space uses another angle to solve the underlying problem. 

## Objective
The purpose of the library is to avoid repetitive tasks, a.k.a DRY (Don't Repeat Yourself), object oritented, a.k.a. OOP, and Monadic, i.e. using the Monada paradigm borrowed from the courses I took in functional programming - Monads and Classes are quite smilar concepts, tend to solve the same issues in computer science, and here they are used altogether to solve our main goals.

## Example
Let's say you work on a data science project in which you are asked to predict prices in the used cars industry. 
And for that purpose you are given some inputs as a csv file with rows and features.

The general strategy or steps is as follows (besides understanding the business needs in details):
1. import major libraries
2. load the file
3. overview the data
4. analyse the data
5. preprocess the data
6. build the model
7. validate the model

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
data = pd.read_csv("Sales.csv")
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

Deep Space library tries to overcome the tedious and repetive tasks and offers a pre-defined set of classes to make it easy and faster:
from DeepSpace.Initialize import Initialize
Initialize(seed=1)
<code>
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
</code>

The >> chevrons are the expression of the Monads strategy.

The above code can also be simplified by creatin a class a not rewriting the whole code gain:
<code>
class SuperOverview(Monad):
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
</code>

Now you can use SuperOverview class like this in your future code :
<code>
    (SuperOverview() >> Milestone())
</code>
