# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sys
sys.path.append(r'E:\Python\introduction_to_ml_with_python-master\introduction_to_ml_with_python-master\mglearn')
import mglearn 


iris_dataset = load_iris() 
print(iris_dataset.keys())


X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'])
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


x = np.array([[1,2,3],[4,5,6]])
print(x)


eye = np.eye(4)
print(eye)
sparse_matrix = sparse.csc_matrix(eye)
print(sparse_matrix)


x = np.linspace(-10,10,100)
y = np.sin(x)
plt.plot(x,y,marker="x")


data = {
        'name':["a","s","d","f"],
        'zxc':["q","w","e","r"],
        'num':[1,2,3,4],
        }
data_pandas = pd.DataFrame(data)
display(data_pandas)
display(data_pandas[data_pandas.num > 1])


iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
grr = pd.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
import numpy as np
X_new = np.array([[5,2.9,1,0.2]])
print(X_new.shape)
prediction = knn.predict(X_new)
print(prediction)
print(iris_dataset['target_names'][prediction])

y_pred = knn.predict(X_test)
print(y_pred)
print(np.mean(y_pred == y_test))
print(knn.score(X_test,y_test))