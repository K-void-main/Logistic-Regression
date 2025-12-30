# %% Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %% Loading Iris Dataset
iris =load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df

# %% Using The Loaded Dataset
df.head()
df.tail()
x = iris.data
y= iris.target
x_train,x_test,y_train,y_test= train_test_split(x,y)

# %% Training Logistic Regression Model
model = LogisticRegression()
model.fit(x_train,y_train)

# %% Making Predictions
model.predict(x_test)


# %%
