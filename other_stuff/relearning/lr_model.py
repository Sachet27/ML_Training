import numpy as np

class LinearRegressor:
    def __init__(self, lr = 0.001):
        self.W= None
        self.lr= lr
        self.__cost_hist= []
        self.__param_hist= []


    def get_weights(self):
        return self.W.flatten()


    def get_cost_hist(self):
        return self.__cost_hist
    

    def _cost(self, y_true, y_hat):
        m= y_true.shape[0]
        J= (0.5/m)* np.sum((y_true - y_hat) **2)
        return J


    def fit(self, X, y, epochs):
        m, n = X.shape
        self.W = np.zeros((n, 1))
        self.__cost_hist = []

        for i in range(epochs):
            y_hat = X @ self.W
            y_delta = y_hat - y
            dJ_dW = (1/m)* (X.T @ y_delta) 
            self.W = self.W - self.lr * dJ_dW

            if i % 10 == 0:
                self.__cost_hist.append(self._cost(y, y_hat))
                self.__param_hist.append(self.W)


    def predict(self, X_test):
        y_pred = X_test @ self.W
        return y_pred
    

