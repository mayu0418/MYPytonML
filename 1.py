# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.datasets import load_iris
iris_dataset = load_iris() 
print(iris_dataset.keys())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'])
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print(x)

from scipy import sparse
eye = np.eye(4)
print(eye)
sparse_matrix = sparse.csc_matrix(eye)
print(sparse_matrix)

%matplotlib inline
import matplotlib.pyplot as plt
x = np.linspace(-10,10,100)
y = np.sin(x)
plt.plot(x,y,marker="x")

import pandas as pd
from IPython.display import display
data = {
        'name':["a","s","d","f"],
        'zxc':["q","w","e","r"],
        'num':[1,2,3,4],
        }
data_pandas = pd.DataFrame(data)
display(data_pandas)
display(data_pandas[data_pandas.num > 1])
