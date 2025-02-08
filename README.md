# Fraud-Detection-System

#**Problem Statement**

- **Description:** Create a classification model to detect fraudulent
transactions. This helps in identifying and preventing fraudulent
activities.
- **Why:** Detecting fraud early can save significant financial losses and
protect customer trust.
- **Tasks:**

    ▪ Gather transaction data.

    ▪ Example datasets Click Here
    
    ▪ Preprocess data (feature engineering, handling imbalanced data).

    ▪ Train classification models (e.g., logistic regression, decision
    trees).

    ▪ Evaluate model performance and implement in a real-time system.

**Using google colab**

from google.colab import files
uploaded = files.upload()

Import required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

Load csv file

# email = pd.read_csv("email_spam.csv")
fraud = pd.read_csv("fraud_data.csv")

# email

fraud.head()

Check the data type of the columns

fraud.info()

Checking whether null values are present

fraud.isnull().sum().sum()

Checking whether duplicates are present

fraud.duplicated().sum()

drop duplicates if present

fraud.drop_duplicates(inplace = True)

fraud.duplicated().sum()

**Checking the outliers are present or not**

for i in fraud.columns:
  if fraud[i].dtypes != 'object':
    plt.boxplot(fraud[i])
    plt.title(i)
    plt.show()

out_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long' ] #Defining the outlier column


Checking the outliers using Violin plot

f, axis = plt.subplots(3,2, figsize=(20,10))
s=sns.violinplot(y=fraud.amt, ax = axis[0, 0])
axis[0, 0].set_title('amt')
s=sns.violinplot(y=fraud.city_pop, ax= axis[0, 1])
axis[0, 1].set_title('city_pop')

s=sns.violinplot(y=fraud.lat, ax = axis[1, 0])
axis[1, 0].set_title('lat')
s=sns.violinplot(y=fraud.long, ax= axis[1, 1])
axis[1, 1].set_title('long')

s=sns.violinplot(y=fraud.merch_lat, ax = axis[2, 0])
axis[2, 0].set_title('merch_lat')
s=sns.violinplot(y=fraud.merch_long, ax= axis[2, 1])
axis[2, 1].set_title('merch_long')


plt.show()

for i in out_cols:
  Q1 = fraud[i].quantile(0.25)
  Q3 = fraud[i].quantile(0.75)
  IQR = Q3 - Q1
  upper_limit = Q3 + 1.5*IQR
  lower_limit = Q1 - 1.5*IQR
  fraud = fraud[(fraud[i]>= lower_limit) & (fraud[i]<= upper_limit)]

# for i in fraud.columns:
#   if fraud[i].dtypes != 'object':
#     plt.boxplot(fraud[i])
#     plt.title(i)
#     plt.show()

# #outlier removal : IQR based
# for i in out_cols:
#   Q1 = BigMart[i].quantile(0.25)
#   Q3 = BigMart[i].quantile(0.75)
#   IQR = Q3 - Q1
#   upper_limit = Q3+1.5*IQR
#   lower_limit = Q1-1.5*IQR
#   BigMart = BigMart[(BigMart[i]>= lower_limit) & (BigMart[i]<=upper_limit)]

# #outlier removal : winsorizing method
# for col in fraud.columns:
#   if fraud[col].dtypes != 'object':
#     p1 = fraud[col].quantile(0.01)
#     p2 = fraud[col].quantile(0.99)
#     fraud[col][fraud[col]<p1]=p1
#     fraud[col][fraud[col]>p2]=p2

for i in fraud.columns:
  if fraud[i].dtypes != 'object':
    plt.boxplot(fraud[i])
    plt.title(i)
    plt.show()

f, axis = plt.subplots(3,2, figsize=(20,10))
s=sns.violinplot(y=fraud.amt, ax = axis[0, 0])
axis[0, 0].set_title('amt')
s=sns.violinplot(y=fraud.city_pop, ax= axis[0, 1])
axis[0, 1].set_title('city_pop')

s=sns.violinplot(y=fraud.lat, ax = axis[1, 0])
axis[1, 0].set_title('lat')
s=sns.violinplot(y=fraud.long, ax= axis[1, 1])
axis[1, 1].set_title('long')

s=sns.violinplot(y=fraud.merch_lat, ax = axis[2, 0])
axis[2, 0].set_title('merch_lat')
s=sns.violinplot(y=fraud.merch_long, ax= axis[2, 1])
axis[2, 1].set_title('merch_long')


plt.show()

Two ways to apply encoding method -
- LabelEncoding
- One-Hot Encoding

**Hence, more column can make ambiguity so applying label encoding to change the data type to int value. Preferred for categorical value.**

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Defining the columns to encode

cols_encode = ['merchant', 'category', 'city',	'state',	'job', 'is_fraud']


for col in cols_encode:
  fraud[col] = le.fit_transform(fraud[col])

fraud

fraud.columns

fraud.drop('trans_date_trans_time', axis = 1, inplace = True)

fraud.drop('dob', axis = 1, inplace = True)

fraud.drop('dob', axis = 1, inplace = True)

fraud.drop('trans_num', axis = 1, inplace = True)

fraud.info()

fraud.describe()

fraud.corr()

plt.figure(figsize=(15,10))
sns.heatmap(fraud.corr(), annot = True, cmap = 'Greens')

X = fraud.drop('is_fraud', axis = 1)
y = fraud['is_fraud']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

d_tree = DecisionTreeClassifier()

d_tree.fit(X_train, y_train)

y_pred = d_tree.predict(X_test)

roc_auc_score(y_pred, y_test)

mean_squared_error(y_pred, y_test)

np.sqrt(mean_squared_error(y_pred, y_test))

mean_absolute_error(y_pred, y_test)

d_tree = DecisionTreeClassifier(max_depth = 4, criterion = 'gini', min_samples_split = 2)
d_tree.fit(X_train, y_train)

from sklearn import tree
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (16, 8))
fig = tree.plot_tree(d_tree, feature_names = X.columns, filled = True)

d_tree.get_depth()  #check the depth of the tree

y_pred_tree = d_tree.predict(X_test)

accuracy_score(y_test, y_pred_tree)

Overfitting

y_pred_train = d_tree.predict(X_train)

accuracy_score(y_train, y_pred_train)

