import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k= 3):
        self.k= k
        self.X_train= None
        self.y_train= None
    
    def fit(self, X_train, y_train):
        self.X_train= X_train
        self.y_train= y_train.flatten()

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x2 - x1)**2))


    def predict(self, X):
        predictions= np.array([self._predict(x) for x in X])
        return predictions.reshape(-1, 1)


    def _predict(self, x):
        distances= [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        knn_indices = np.argsort(distances)[:self.k]
        knn= self.y_train[knn_indices]
        most_common_label= Counter(knn).most_common(1)[0][0]
        return most_common_label
    
class Encoder:
    def __init__(self):
        self.categories= None
        self.cats_to_index= None

    def fit(self, vec):
        vector= vec.flatten()
        self.categories= np.unique(vector)
        self.cats_to_index= {cat: i for i, cat in enumerate(self.categories)}

    def transform(self, vec):
        vector= vec.flatten()
        encoded_vector= [self.cats_to_index[el] for el in vector]
        return np.array(encoded_vector).reshape(-1, 1)   
    
    def fit_transform(self, vec):
        self.fit(vec)
        return self.transform(vec)