#!/usr/bin/python

#Version checks and import statements
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

import pickle

import pandas
from pandas import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# To load the model from disk
loaded_model = pickle.load(open('machinelearningmodel.sav', 'rb'))
#result = loaded_model.score(X_train, Y_train)
#print(result)
print(loaded_model)
knn = KNeighborsClassifier()
#knn.fit(loaded_model, Y_train)
knn.fit(loaded_model, 'class')
predictions = knn.predict([[3398,4,2,1,1,2,1,0,0,3,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,4,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,11,0,0,0,0,2,11,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,3,0,1,1,0,2,0,0,0,0,0,0,0,0,1,0,0,0,4,0,0,0,2,1,0,2,0,0,0,0,0,0,1,0,0,0,1,2,2,0,2,0,0,3,0,0,2,0,0,0,6,0,0,2,0,0,0,0,3,1,0,3,2]])
print(knn.predict([[3398,4,2,1,1,2,1,0,0,3,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,4,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,11,0,0,0,0,2,11,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,3,0,1,1,0,2,0,0,0,0,0,0,0,0,1,0,0,0,4,0,0,0,2,1,0,2,0,0,0,0,0,0,1,0,0,0,1,2,2,0,2,0,0,3,0,0,2,0,0,0,6,0,0,2,0,0,0,0,3,1,0,3,2]]))

