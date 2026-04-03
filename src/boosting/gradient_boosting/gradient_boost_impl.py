import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators= 100, learning_rate= 0.1, min_sample_split= 2, max_leaf_count= 32, n_features= "log2"):
        self.n_estimators= n_estimators
        self.learning_rate= learning_rate
        self.min_sample_split= min_sample_split
        self.n_features= n_features
        self.max_leaf_count= max_leaf_count 
        self.__pseudo_residuals= None
        self._estimators= []
        self.__y_initial= 0.0 

    
    def _build_decision_tree(self, X):
        dt= DecisionTreeRegressor(
                max_leaf_nodes= self.max_leaf_count, 
                min_samples_split= self.min_sample_split,
                max_features= self.n_features
                )
        
        dt.fit(X, self.__pseudo_residuals)

        y_preds= dt.predict(X)
        return dt, y_preds        


    
    def fit(self, X, y):
        f_x= np.mean(y)
        self.__y_initial = f_x

        for i in range(self.n_estimators):
            self.__pseudo_residuals= y - f_x  
            dt, y_pred= self._build_decision_tree(X)
            
            self._estimators.append(dt)
            f_x= f_x + self.learning_rate * y_pred  

        


    def predict(self, X):
        y_preds= np.full(X.shape[0], self.__y_initial)
        for model in self._estimators:
            y_preds+= (self.learning_rate * model.predict(X))
        return y_preds
        


    
