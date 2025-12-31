import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

class StandardScaler():
    def __init__(self):
        self.mean= 0
        self.std= 1
        self.has_zero_std= None
    
    def fit(self, arr):
        #arr is a 2d matrix 
        self.mean= np.mean(arr, axis= 0) 
        self.std= np.std(arr, axis= 0)
        
        #handling zero std
        self.has_zero_std= (self.std == 0)
        self.std[self.has_zero_std]= 1.0

        return self.mean, self.std
    

    def transform(self, arr):
        norm_arr= (arr - self.mean)/ self.std
        return norm_arr
    
    def detransform(self, arr_norm):
        orig_arr= arr_norm* self.std + self.mean
        return orig_arr


class L2LinearRegression():
    def __init__(self):
        self.weights= None
        self.bias= 0
        self.__cost_hist= []

    def __compute_cost(self, y_train, y_hat, lamda, weights):
        m= y_train.shape[0]
        J= (0.5/m)* np.sum((y_hat- y_train)**2) + (0.5/m)* lamda* np.sum(weights **2)
        return J


    def fit(self, X, y, alpha= 0.0001, epochs= 20000, lamda= 0.1, cost_hist_frequency= 5):
        m= X.shape[0]
        X_train= np.c_[np.ones(m) , X.copy()]
        y_train= y.copy()
        W_b= np.zeros((X_train.shape[1] , 1))   

        for i in range(epochs):
            y_hat= X_train @ W_b
            
            if i % cost_hist_frequency ==0:
                J= self.__compute_cost(y_train, y_hat, lamda, W_b[1:].flatten())
                self.__cost_hist.append(J)

            dJ_dW_b= (1/m)* (X_train.T @ (y_hat - y_train))

            # for regularization
            dJ_dW_b[1:]+= ((lamda/m)* W_b[1:])

            W_b-= alpha* dJ_dW_b
        
        self.weights= W_b[1:].flatten()
        self.bias= W_b[0][0]
    
    
    def get_cost_hist(self):
        return self.__cost_hist.copy()
    
    def predict(self, X):
        X_test= np.c_[np.ones(X.shape[0]) ,X.copy()]
        W_b= np.insert(self.weights, 0, self.bias).reshape(-1, 1)
        y_pred= X_test @ W_b
        return y_pred                    
    
class OneHotEncoder():
    def __init__(self):
        self.categories= None
        self.cats_to_index = None
    
    def fit(self, y):
        # shape must be (n, 1)
        arr= y.flatten()
        self.categories= np.unique(arr)
        self.cats_to_index = {cat: idx for idx, cat in enumerate(self.categories)}
        return self.categories, self.cats_to_index

    def transform(self, y):
        arr= y.flatten()
        n= len(self.categories)
        y_encoded= np.zeros((len(arr) , n))

        for idx, cat in enumerate(arr):
            y_encoded[idx , self.cats_to_index[cat]] = 1
    
        return y_encoded 