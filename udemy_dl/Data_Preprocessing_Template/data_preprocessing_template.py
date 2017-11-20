# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis = 0)
X[:,1:3] = imputer.fit(X[:,1:3]).transform(X[:,1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Transform categorical values to numerical values between 0 and length(class)-1
le_X = LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])
# Since X[0] has categorical values, so we want to transform numerical values to attributes has only 0 and 1

# Transform categorical to dummies?
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray();

# Transform y
le_y = LabelEncoder()
y = le_y.fit_transform(y)


# Splitting the database into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# Feature Scaling: Transform attribute values so all attributes would have same impact on the result. Two methods: Standardisation(between -1 to 1) and Normalisation(between 0 to 1). 

# Standardisation
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # do not use fit_transform. Make sure train and test fit to same model.

# No feature scaling on y because its a binary classification problem.




# Basic Steps:

# 1. Import libraries
# 2. Import dataset (Optional: fix missing data/Encoding categocial data)
# 3. Splitting dataset into training and testing sets
# 4. Feature Scaling