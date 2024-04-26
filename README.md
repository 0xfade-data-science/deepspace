<H1> Deep Space </H1>

Deep Space is an umbrella library that runs on top of major data science libraries as <a href="https://scikit-learn.org/"> scikit-learn learn </a> and <a href="https://www.statsmodels.org/">statsmodel<a> to site only few.
It is the result of the course I took from MIT Education Professional labeled MIT Advanced Data Science Program.

<h2>Objective</h2>
The purpose of the library is to avoid repetitive tasks, a.k.a DRY (Don'tRepeat Yourself), object oritented, a.k.a.OOP, and Monadic, i.e. using the Monada paradigm borrowed from the courses I took in functional programming - Monads and Classes are quite smilar concepts, tend to solve the same issues in computer science, and here they are used together to solve our main goals.

<h2>Example</h2>
Let's say work on a data science project in which you are asked to predict prices in the used cars industry. 
And for that objective you are given some inputs as a csv file with rows and columns (or features).

The general strategy is as follows (besides understanding the business needs in details):
1. import major libraries
2. load the file
3. overview the data
4. analyse the data
5. preprocess the data
6. build the model
7. validate the model

It goes like this :
<code>
#Libraries for reading and manipulating data
import numpy as np
import pandas as pd

#Import data viz libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for building linera regression model
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Import library for pre processing data
from sklearn.model_selection import train_test_split

</code>


