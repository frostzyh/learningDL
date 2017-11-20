#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:35:42 2017

@author: yehuizhang
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # Encode Geography
labelencoder_X_geo = LabelEncoder()
X[:, 1] = labelencoder_X_geo.fit_transform(X[:, 1])
    # Encode Gender
labelencoder_X_sex = LabelEncoder()
X[:, 2] = labelencoder_X_sex.fit_transform(X[:, 2])
    # Create dummy variables for Geography
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# To avoid dummy variable trap, drop one dummy variable.
# Reference: http://www.algosome.com/articles/dummy-variable-trap-regression.html
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)












#---------Trail 1---------

# BUild ANN

# Initializing ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
    # To choose the number of nodes in the first hidden layer, you can, (Not as a rule of thumb) use this formular 1/2(# of attributes in input + # of outputs)
    # input_shape: specify the shape of initial inputs for the first hidden layer(the input nodes)
    # Use Rectifier (relu) as activation fuction.
    # kernel_initializer is for initializing weights to small numbers close to 0 but not 0.
classifier.add(Dense(6, input_shape = (11,) , activation = 'relu', kernel_initializer='glorot_uniform'))


# Add second Hidden layer
classifier.add(Dense(6, activation = 'relu', kernel_initializer='glorot_uniform'))

# Add the output layer. Use sigmoid for binary output or softmax for multiple output nodes. 
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer='glorot_uniform'))

# Compiling the ANN
    # Optiizer: includes SGD, Adam, Adagrad and so on.
    # Loss function: use logarithmic loss: binary_crossentropy for binary outcome or categorical_crossentropy for three or more outcomes.
classifier.compile('Adam', 'binary_crossentropy', metrics = ['accuracy'])

# Fit the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Convert probability values in y_pred to either 0 or 1
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

test_accuracy = (cm[0][0] + cm[1][1]) / cm.sum()



# ANN Algorithm(6 Steps)

    # S1: Randomly initialize the weights to small numbers close to 0 (but not 0)
    # S2: Input the first observation of your dataset in the input layer, each feature in one input node
    # S3: Forward-Propagation: from left to right. Activation Functions: Use Rectifier for hidden layer and sigmoid function for output layer since sigmoid function gives probability of confidence of the output.
    # S4: Compare y_predicted to y_actual. Measure the error/cost
    # S5: Back propagation: From right to left. Update weights according to the error. 
    # S6: Repeat Step 2 to 5 using either Reinforcement learning(update weight after each observation) or batch Learning(update weights after a batch of observation)
    
    
    
# Single input prediction:
# France, 600, Male, 40, 3, 60000, 2, Yes, Yes, 50000
# Use double bracket [[]] to create 2D array.
single_test = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
single_test = sc.transform(single_test)
classifier.predict(single_test) > 0.5






#---------Trail 2---------

#Evaluation the RNN
# https://keras.io/scikit-learn-api/
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, input_shape = (11,) , activation = 'relu', kernel_initializer='glorot_uniform'))
    classifier.add(Dense(6, activation = 'relu', kernel_initializer='glorot_uniform'))
    classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer='glorot_uniform'))
    classifier.compile('Adam', 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# 10-fold cross validation. (cv = 10). Use all cpus (n_jpbs = -1)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()        # 0.859499993771
variance = accuracies.std()     # 0.0100187317888



#---------Trail 3---------
# Improving the ANN
# Dropout regularization to reduce overfitting if needed
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(6, input_shape = (11,) , activation = 'relu', kernel_initializer='glorot_uniform'))
classifier.add(Dropout(rate = 0.1))  # disable 10% nodes in this layer. If rate =1, the network learns nothing.

classifier.add(Dense(6, activation = 'relu', kernel_initializer='glorot_uniform'))
classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer='glorot_uniform'))
classifier.compile('Adam', 'binary_crossentropy', metrics = ['accuracy'])


#---------Trail 4---------
# Parameter Tunning
# Let program to find optimal parameters
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform', input_shape = (11,)))
    classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer='glorot_uniform'))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer='glorot_uniform'))
    classifier.compile(optimizer=optimizer, loss ='binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier) #build_fn should construct, compile and return a Keras model,
parameters = {'batch_size':[25, 32],
              'epochs':[100,500],
              'optimizer':['Adam','RMSprop']}

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_





# ****************************** Results ******************************

# Before Tunning
# 10-Fold Cross-validation  Accuracy: 0.85949  V: 0.01001



# Parameter Tuning
# Best result: Accuracy: 0.86237  {'batch_size': 25, 'epochs': 500, 'optimizer': 'RMSprop'}




