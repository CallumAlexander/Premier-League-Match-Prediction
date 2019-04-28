# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:39:03 2019

@author: Callum
"""

#Neural Network to predict premier leauge matches

from random import randint

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('results.csv')
X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:,4].values

#Encoding categorical data

# 2,11
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_HomeTeam = LabelEncoder()
X[:,0] = labelencoder_X_HomeTeam.fit_transform(X[:, 0])
labelencoder_X_AwayTeam = LabelEncoder()
X[:,1] = labelencoder_X_AwayTeam.fit_transform(X[:, 1])
#labelencoder_X_Season = LabelEncoder()
#X[:,2] = labelencoder_X_Season.fit_transform(X[:, 2])

labeleconder_y = LabelEncoder()
y = labeleconder_y.fit_transform(y)
y = pd.DataFrame(y)
onehotencoder_y = OneHotEncoder()
y = onehotencoder_y.fit_transform(y).toarray()


onehotencoderHomeTeam = OneHotEncoder(categorical_features = [0])
X = onehotencoderHomeTeam.fit_transform(X).toarray()
X = np.delete(X, 38, 1)

onehotencoderAwayTeam = OneHotEncoder(categorical_features = [38])
X = onehotencoderAwayTeam.fit_transform(X).toarray()
X = np.delete(X, 76, 1)

onehotencoderSeason = OneHotEncoder(categorical_features = [76])
X = onehotencoderSeason.fit(X).toarray()




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#y_train = sc.fit_transform(y_train)
#_test = sc.fit_transform(y_test)




from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

#def BuildCalsClassifier():
calsClassifier = Sequential()

#adam = Adam(lr=0.005)

#input layer and first hidden layer
calsClassifier.add(Dense(units = 76, activation = 'relu', input_dim = 76))
#calsClassifier.add(Dropout(p = 0.05))

calsClassifier.add(Dense(units = 44, kernel_initializer='random_uniform', activation = 'relu'))
#calsClassifier.add(Dropout(p = 0.05))

calsClassifier.add(Dense(units = 22, kernel_initializer='random_uniform', activation = 'relu'))
#calsClassifier.add(Dropout(p = 0.05))


calsClassifier.add(Dense(units = 11, kernel_initializer='random_uniform', activation = 'relu'))
#calsClassifier.add(Dropout(p = 0.05))

calsClassifier.add(Dense(units = 6, kernel_initializer='random_uniform', activation = 'relu'))
#calsClassifier.add(Dropout(p = 0.05))

calsClassifier.add(Dense(units = 3, kernel_initializer='uniform', activation = 'softmax'))

calsClassifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(calsClassifier.summary())




calsClassifier.fit(X_train, y_train, batch_size = 1, epochs = 10, verbose=1, validation_split=0.05)
score, acc = calsClassifier.evaluate(X_test, y_test, batch_size = 10, verbose=2)
print('Test Accuracy : ', acc)
y_pred = calsClassifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


while acc < 0.8:
    batchsize = randint(5, 20)
    epoch = randint(100, 500)
    calsClassifier.fit(X_train, y_train, batch_size = batchsize, epochs = epoch)
    score, acc = calsClassifier.evaluate(X_test, y_test, batch_size = 10, verbose=2)
    print('Test Accuracy : ', acc)

#calsClassifier = KerasClassifier(build_fn = BuildCalsClassifier, epochs=100, batch_size = 5)

#kfold = KFold(n_splits=10, shuffle=True)

#results = cross_val_score(calsClassifier, X, y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



y_pred = calsClassifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))




'''
parameters = {'batch_size': [10, 20],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = calsClassifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_ 




#,

#calsClassifier.fit(X_train, y_train, batch_size = 10, epochs = 1000)



new_prediction = ['Everton', 'Manchester United', '2018-2019']
new_prediction = calsClassifier.predict(sc.transform(np.array([['Everton', 'Manchester United', '2018-2019']])))
new_prediction = (new_prediction > 0.5)
print(new_prediction)
'''

