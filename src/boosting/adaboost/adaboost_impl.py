from weighted_dt import DecisionTreeClassifier
import numpy as np
from mlxtend.plotting import plot_decision_regions


class AdaBoostClassifier:
    def __init__(self, n_estimators= 50, learning_rate= 1.0):
        self.n_estimators= n_estimators
        self.learning_rate= learning_rate
        self._estimators= []
        self._weights= None
        self._alphas= []
        
        # for labelling classes as -1 or 1
        self.classes_= None
        self._class_map= None
        self._inv_class_map= None        
    
    
    def _total_error(self, y_true, y_preds):
        return np.sum(self._weights[y_preds != y_true])


    def _build_decision_stump(self, X, y):
        dt= DecisionTreeClassifier(max_depth= 1, sample_weights= self._weights, min_samples_split= 2)
        dt.fit(X, y)

        y_preds= dt.predict(X)
        total_error= self._total_error(y, y_preds)
        return dt, y_preds, total_error


    
    def fit(self, X, y):
        self._estimators = []
        self._alphas = []
        
        
        n_samples, n_features= X.shape
        self._weights= np.ones(n_samples) / n_samples
        
        #creating labels -1 and 1 for y
        self.classes_= np.unique(y)
        self._class_map= {
            self.classes_[0]: -1,
            self.classes_[1]: 1
            }
        
        self._inv_class_map= {
            -1: self.classes_[0],
            1: self.classes_[1]
            }


        X_train, y_train= X, np.vectorize(self._class_map.get)(y)


        for i in range(self.n_estimators):
            #training decision stump
            dt, y_preds, total_error, = self._build_decision_stump(X_train, y_train)

            if total_error > 0.5:
                continue

            total_error = np.clip(total_error, 1e-10, 1-1e-10)

            self._estimators.append(dt)

            alpha=  self.learning_rate * 0.5* np.log((1 - total_error) / (total_error + 1e-10))
            self._alphas.append(alpha) 

            #updating and normalizing sample weights
            self._weights= np.where(y_preds == y_train, 
                                   self._weights * np.exp(-alpha),
                                   self._weights * np.exp(alpha))
            self._weights= self._weights / np.sum(self._weights)





    def predict(self, X):
        #each row is a model and column is all predictions for a sample
        model_wise_predictions= np.array([model.predict(X) for model in self._estimators])

        #each row is all predictions for a sample and column is model
        sample_wise_predictions= np.transpose(model_wise_predictions)

        res= np.sum(sample_wise_predictions * self._alphas, axis= 1)
        y_preds= np.where(res < 0, -1, 1)

        return np.vectorize(self._inv_class_map.get)(y_preds)
