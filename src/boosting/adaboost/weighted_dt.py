import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_idx= None, threshold= None, left= None, right= None, value= None):
        #for decision nodes
        self.feature_idx= feature_idx
        self.threshold= threshold
        self.left= left
        self.right= right


        #for leaf
        self.value= value
        


class DecisionTreeClassifier:
    def __init__(self, n_features= None, max_depth= 100, min_samples_split= 3, sample_weights= None):
        self.root= None
        self.n_features= n_features
        self.max_depth= max_depth
        self.min_samples_split= min_samples_split
        self.sample_weights= sample_weights

    
    def gini_reduction(self, left_y, right_y, left_w, right_w, parent_y, parent_w):
        total_w= np.sum(parent_w)
        left_weight= np.sum(left_w) / total_w 
        right_weight= np.sum(right_w)/ total_w
        return self._gini_impurity(parent_y, parent_w) - (left_weight * self._gini_impurity(left_y, left_w) + right_weight* self._gini_impurity(right_y, right_w))


    def _gini_impurity(self, y, weights):
        total_w= np.sum(weights)
        labels= np.unique(y)
        p= np.array([np.sum(weights[y== label])/total_w for label in labels])
        gini= 1- np.sum(p**2)
        return gini
    

    def build_tree(self, data, weights, curr_depth= 0):
        X, y = data[:, :-1], data[:, -1]

        n_samples, n_features= X.shape

        if curr_depth < self.max_depth and n_samples>= self.min_samples_split:
            n_feats= self.n_features if self.n_features is not None else n_features
            n_feats= min(n_feats, n_features)

            best_split= self._best_split(data, weights, n_feats)
            
            if best_split['gini_reduction'] > 0:
                left_data= best_split['left_data']
                right_data= best_split['right_data']
                left_w= best_split['left_w']  
                right_w= best_split['right_w']

                left= self.build_tree(left_data, left_w, curr_depth + 1) 
                right= self.build_tree(right_data, right_w, curr_depth + 1)

                return Node(
                    feature_idx= best_split['feature_idx'],
                    threshold= best_split['threshold'],
                    left= left,
                    right= right
                ) 
        

        #return node with highest weight contribution
        labels= np.unique(y)
        weighted_sums= [np.sum(weights[y== label]) for label in labels] 
        max_index= np.argmax(weighted_sums)


        return Node(
            value= labels[max_index]
        )
    


    
    def _best_split(self, data, weights, n_feats):
        optimal_split= {'feature_idx': None, 'threshold': None, 'gini_reduction': -1, 'left_data': None, 'right_data': None, 'left_w': None, 'right_w': None}

        total_n_features = data.shape[1] - 1 #total no. of features to take random indices
        feat_idxs= np.random.choice(total_n_features, n_feats, replace= False) # take n_feats elements from total_n_features choices

        for feature_idx in feat_idxs:
            unique= np.unique(data[:, feature_idx])
            thresholds= (unique[1:] + unique[:-1] ) / 2
            
            for threshold in thresholds:
                left_data, right_data, left_w, right_w = self.__split(data, weights, feature_idx, threshold)

                if len(left_data)> 0 and len(right_data) > 0:
                    parent_y, left_y, right_y = data[:, -1], left_data[:, -1], right_data[:, -1]
                    gini_red= self.gini_reduction(left_y, right_y, left_w, right_w, parent_y, weights)
                    
                    if gini_red > optimal_split['gini_reduction']:
                        optimal_split['feature_idx']= feature_idx
                        optimal_split['threshold']= threshold
                        optimal_split['left_data']= left_data
                        optimal_split['right_data']= right_data
                        optimal_split['left_w']= left_w
                        optimal_split['right_w']= right_w
                        optimal_split['gini_reduction']= gini_red

        return optimal_split
        


    def __split(self, data, weights, feature_idx, threshold):
        left_mask= data[:, feature_idx] <= threshold 
        right_mask= data[:, feature_idx] > threshold 

        left_data= data[left_mask]
        right_data= data[right_mask]
        left_w= weights[left_mask]
        right_w= weights[right_mask]

        return left_data, right_data, left_w, right_w        



    def fit(self, X, y):
        if self.sample_weights is None:
            self.sample_weights = np.ones(X.shape[0])

        data= np.concatenate([X, y.reshape(-1, 1)], axis= 1)
        
        self.root= self.build_tree(data, self.sample_weights)

        

    def predict(self, X):
        preds= [self._predict_class(x, self.root) for x in X]
        return preds
    

    def _predict_class(self, x, node):
        if node.value is not None:
            return node.value
        
        feature_idx, threshold= node.feature_idx, node.threshold
        if x[feature_idx] <= threshold:
            return self._predict_class(x, node.left)
        else:
            return self._predict_class(x, node.right)
        