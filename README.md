# Deep Space

Deep Space is an umbrella library that runs on top of major data science libraries as [scikit-learn](https://scikit-learn.org/), [statsmodel](https://www.statsmodels.org/), [tensorflow](https://www.tensorflow.org/) to site only a few.
It is the result of the 4 months course I took from **MIT Education Professional** labeled **[MIT Advanced Data Science Program](https://professional-education-gl.mit.edu/mit-applied-data-science-course)**.

In some aspects this Deep Space library resembles the Pipeline library, which I discoverd quite late during the program but Deep Space uses another angle to solve the underlying problem.
You can see it as a **syntactic sugar** with a bunch of **reusable classes**.

## Objective

The purpose of the library is to avoid repetitive tasks, a.k.a DRY (Don't Repeat Yourself), object oriented, a.k.a. OOP, and Monadic, i.e. using the Monad paradigm borrowed from functional programming world (Haskel for instance) - Monads and Classes are quite similar concepts, tend to solve the same issues in computer science, and here they are used altogether to solve our main goals.

Using OOP paradigm, DeepSpace will **encapsulate** x, y, x_train, x_test, y_train, y_test - as well as other variables - making it easy to manage them.

With Milestone objects, integrated cloning, and checkpoint backup, it makes it easier to go to a specific checkpoint in the project and start from there a new branch.

DeepSpace will come with a bunch of predefined classes hence the programmer does not need to create the wheel again.

DeepSpace library will simplify a first analysis of the problem, it gives a rapid glimpse thanks to it's packaged classes.

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
6. build models
7. validate models

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
from deepspace.Initialize import Initialize
Initialize(seed=1)
#..

file = "travel_train_data"
#overview of the data
_ =  (
     >> CSVLoader(file, ",")
     >> Overview()
     >> Describe()
     >> CheckUniqueness()
     >> CheckNulls()
     >> CheckDuplicated()
     >> CheckOutliers()
)
```

The >> chevrons are the expression of the Monads strategy.

The above code can also be simplified by creating a class and not rewriting the whole code again:

```
class SuperOverview(Transformer):
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
    )
```

Now you can use SuperOverview class like this in your future code :

```
_ =  (SuperOverview(file) >> Milestone())
```

#### Data analysis, or Exploratory Data Analysis (EDA)

Just like so to show a bunch of plots (histograms, boxplots, etc) and calculations for the numerical features:

```
    (_ >> numerical.Univariate(num_cols=[]))
```

You can set the num_cols parameter to only show specific featrues.

And the below code is for categorical features:

```
    (_ >> categorical.Univariate() )
```

The same hereafter for bivariate analysis:

```
    (_ >> Bivariate(num_cols=(), cat_cols=[]))
```

Anf if you want to see the heatmap only :
The same hereafter for bivariate analysis:

```
    (_ >> Heatmap(num_cols=()) )
```

#### Data preprocessing

##### Drop non useful features

Drop ID feature:

```
_ = (_
    >> DropAdjuster(['ID'])
    >> DropCols()
)
```

Drop ID and Price_OLD feature:

```
_ = (_
    >> DropAdjuster(['ID','Price_OLD'])
    >> DropCols()
)
```

##### Null imputation and dupàlicates removal

Impute nulls (by default categorical with be imputed with the mode and numerical with the mean):

```
saveto = 'milestone-imputation-no-dupes.pkl'
_ = (_
    >> ProcessImputation()
    >> DropDuplicates()
    >> Save(saveto)
)
```

As you can see here can also save the result to a file (pckle file) to be reused later.

##### Feature engineering

```
_ = (_
    >> Log('Departure_Delay_in_Mins', 'Departure_Delay_in_Mins_Log')
    >> Log('Arrival_Delay_in_Mins', 'Arrival_Delay_in_Mins_Log')
    >> Log('Travel_Distance', 'Travel_Distance_Log')
    >> Sum(['Arrival_Delay_in_Mins', 'Departure_Delay_in_Mins'], 'X_Delay_Sum')
    >> Log('X_Delay_Sum', 'X_Delay_Sum_Log', 10)
    >> Polynom(['Departure_Delay_in_Mins_Log', 'Arrival_Delay_in_Mins_Log'], 'X_Delay_Ploy')
    >> Sine('Travel_Distance', 'Travel_Distance_Sine')
)
```

##### Separating and Splitting

```
_ = (_
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
)
```

##### Model Building and testing

```
_ = (_
        >> Load('clean_train_chain_result-post-merge-split.pkl')
        >> LogisticRegression()
        #>> Debug()
        >> PerfCalculator()
        >> ShowCoeffs()
        >> ShowInvCoeffs()
)
```
### Complete example
You can find a complete example here: [Uber example notebook](https://github.com/0xfade-data-science/deepspace/blob/main/examples/data%20exploration/Uber/Uber_Case_Study-the-deepspace-way.ipynb)

## Versions

Currently the library is tested with this configuration:
|-|version|
| --- | --- |
|python | 3.9.18|
|pandas | 2.2.1|
|numpy | 1.26.4|
|tensorflow | 2.16.1|
|sklearn | 1.4.2|
|statsmodels | 0.14.0|
