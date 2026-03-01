import numpy as np
from collections import Counter
from randomized_dtc import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_estimators= 100, min_samples_split= 5, max_depth= 100, n_features= None):
        self.n_estimators= n_estimators
        self.min_samples_split= min_samples_split
        self.max_depth= max_depth
        self.n_features= n_features
        self.estimators= []

    def _bootstrap_sample(self, X, y):
        n_samples= X.shape[0]
        idxs= np.random.choice(n_samples, n_samples, replace= True)
        return X[idxs], y[idxs]
        
    
    def fit(self, X, y):
        self.estimators= []
        if self.n_features is None:
            self.n_features= int(np.sqrt(X.shape[1]))

        for _ in range(self.n_estimators):
            X_boot, y_boot= self._bootstrap_sample(X, y)

            tree= DecisionTreeClassifier(min_samples_split= self.min_samples_split, 
                                         max_depth= self.max_depth, 
                                         n_features= self.n_features)
            tree.fit(X_boot, y_boot)

            self.estimators.append(tree)
    
    
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    
    def predict(self, X):
        #each row is all predictions from a particular tree
        model_wise_predictions= np.array([tree.predict(X) for tree in self.estimators])

        #each row is all predictions about a particular sample   
        sample_wise_predictions= np.transpose(model_wise_predictions) 

        y_preds= np.array([self._most_common_label(pred) for pred in sample_wise_predictions])
        return y_preds
