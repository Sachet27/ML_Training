import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_idx= None, threshold= None, info_gain= None, left= None, right= None, value= None):
        #decision node
        self.feature_idx= feature_idx
        self.threshold= threshold
        self.info_gain= info_gain
        self.left= left
        self.right= right
        
        #leaf node
        self.value= value

    def is_leaf_node(self):
        return self.value is not None
    

class DecisionTreeClassifier:
    def __init__(self, min_samples_split, max_depth, n_features):
        self.min_samples_split= min_samples_split
        self.max_depth= max_depth
        self.n_features= n_features
        self.root= None


    def build_tree(self, data, curr_depth= 0):
        X, y= data[:, :-1], data[:, -1]
        n_samples, n_feats= X.shape

        if n_samples >= self.min_samples_split and curr_depth < self.max_depth and len(np.unique(y)) > 1:
            n_features= self.n_features if (self.n_features <= n_feats or self.n_features is None) else n_feats 
            random_feat_idxs= np.random.choice(n_feats, n_features, replace= False)
            best_split= self._best_split(data, random_feat_idxs)

            if best_split['info_gain'] > 0:
                left= self.build_tree(best_split['left_data'], curr_depth + 1)
                right= self.build_tree(best_split['right_data'], curr_depth + 1)
                
                return Node(
                    feature_idx= best_split['feat_idx'],
                    threshold= best_split['threshold'],
                    info_gain= best_split['info_gain'],
                    left= left,
                    right= right
                )        

        return Node( 
            value= Counter(y).most_common(1)[0][0]
            ) 
    
    

    def _best_split(self, data, feat_idxs):
        optimal_split= {'feat_idx': None, 'threshold': None, 'info_gain': -1, 'left_data': None, 'right_data': None}
        parent_y= data[:, -1]
        parent_entropy= self._entropy(parent_y)

        for idx in feat_idxs:
            #thresholds will be midpoints between unique points
            unique_vals= np.unique(data[:, idx]) 
            thresholds= (unique_vals[1:] + unique_vals[:-1])/2 

            for threshold in thresholds:
                left_data, right_data = self._split(data, idx, threshold)

                if len(left_data) > 0 and len(right_data) > 0: 
                    left_y, right_y = left_data[:, -1], right_data[:, -1]
                    
                    info_gain= self._information_gain(left_y, right_y, parent_entropy)
                    if info_gain > optimal_split['info_gain']:
                        optimal_split['feat_idx'] = idx
                        optimal_split['threshold'] = threshold
                        optimal_split['info_gain'] = info_gain
                        optimal_split['left_data'] = left_data
                        optimal_split['right_data'] = right_data
        
        return optimal_split
    


    def _entropy(self, y):
        labels= np.unique(y)
        entropy= 0

        for label in labels:
            p= len(y[y == label]) / len(y)
            entropy+= (-p * np.log2(p))
        
        return entropy
    

    
    def _information_gain(self, left_y, right_y, parent_entropy):
        total= len(left_y) + len(right_y)
        left_weight= len(left_y) / total
        right_weight= len(right_y) / total

        return parent_entropy - (left_weight * self._entropy(left_y) + right_weight * self._entropy(right_y))
    


    def _split(self, data, feature_idx, threshold):
        left_data= data[data[:, feature_idx] <= threshold]
        right_data= data[data[:, feature_idx] > threshold]
        return left_data, right_data



    def fit(self, X, y):
        dataset= np.concatenate([X, y], axis= 1)
        self.root= self.build_tree(dataset)


    def predict(self, X):        
        y_preds= np.array([self._predict_class(row, self.root) for row in X])
        return y_preds
    

    def _predict_class(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        feat_idx, threshold= node.feature_idx, node.threshold 
        if x[feat_idx] <= threshold:
            return self._predict_class(x, node.left)
        else:
            return self._predict_class(x, node.right)



