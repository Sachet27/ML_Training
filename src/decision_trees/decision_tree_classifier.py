import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_idx= None, threshold= None, info_gain= None, value= None, left= None, right= None):
        # decision node
        # self.data= data
        self.feature_idx= feature_idx
        self.threshold= threshold
        self.info_gain= info_gain
        self.left= left
        self.right= right

        #leaf value
        self.value= value


class DecisionTree:
    def __init__(self, min_samples= 6, max_depth= 3):
        self.min_samples= min_samples
        self.max_depth= max_depth
        self.root= None


    def _entropy(self, y):
        labels= np.unique(y)
        entropy= 0

        for label in labels:
            p= len(y[y == label]) / len(y) 
            entropy+= (-p * np.log2(p)) 
        return entropy
    

    def information_gain(self, left_y, right_y, parent_y):
        left_weight= len(left_y)/ len(parent_y)
        right_weight= len(right_y)/ len(parent_y)
        return self._entropy(parent_y) - (left_weight* self._entropy(left_y) + right_weight* self._entropy(right_y))



    def build_tree(self, data, curr_depth= 0):
        X, y= data[:, :-1], data[:, -1]
        n_samples, n_features= X.shape
        n_unique_labels= np.unique(data[:, -1])

        if n_samples >= self.min_samples and curr_depth < self.max_depth and n_unique_labels> 1:
            best_split = self.best_split(data, n_features) 

            if best_split['info_gain'] > 0:
                left= self.build_tree(best_split['left_dataset'], curr_depth+ 1) 
                right= self.build_tree(best_split['right_dataset'], curr_depth + 1)
                return Node(
                    feature_idx= best_split['feature_idx'], 
                    threshold= best_split['threshold'], 
                    info_gain= best_split['info_gain'],
                    left= left, 
                    right= right
                    )
        
        return Node(value= Counter(y).most_common(1)[0][0])



    def best_split(self, data, n_features):
        optimal_split= {'feature_idx': None, 'threshold': None, 'info_gain': -1, 'left_dataset': None, 'right_dataset': None}

        for feature_idx in range(n_features):
            thresholds = np.unique(data[:, feature_idx])
            for threshold in thresholds:
                left_data, right_data = self._split(data, feature_idx, threshold)
                if len(left_data) > 0 and len(right_data)> 0:
                    left_y, right_y, parent_y = left_data[:, -1], right_data[:, -1], data[:, -1]
                    info_gain= self.information_gain(left_y, right_y, parent_y)

                    if info_gain > optimal_split['info_gain']:
                        optimal_split['info_gain']= info_gain
                        optimal_split['feature_idx']= feature_idx
                        optimal_split['threshold']= threshold
                        optimal_split['left_dataset'] = left_data
                        optimal_split['right_dataset'] = right_data

        return optimal_split



    def _split(self, data, feature_idx, threshold):
        left_data= data[data[:, feature_idx] <= threshold ]
        right_data= data[ data[:, feature_idx] > threshold ]

        return left_data, right_data
    

    def fit(self, X, y):
        data= np.concatenate([X, y.reshape(-1, 1)], axis= 1)
        self.root= self.build_tree(data)


    def predict(self, X):
        def predict_class(row, node: Node):
            if node.value != None:
                return node.value

            feature_idx, threshold= node.feature_idx, node.threshold
            if row[feature_idx] <= threshold:
                return predict_class(row, node.left)
            else:
                return predict_class(row, node.right)
        
        y_pred = np.array([predict_class(row, self.root) for row in X]) 
        return y_pred
