import numpy as np

class Node:
    def __init__(self, feature_idx= None, threshold= None, var_reduction= None , left= None, right= None, value= None):
        # decision nodes
        self.feature_idx= feature_idx
        self.threshold= threshold
        self.var_reduction= var_reduction
        self.left= left
        self.right= right

        #leaf node
        self.value= value


    def is_leaf_node(self):
        return self.value is not None



class DecisionTreeRegressor:
    def __init__(self, min_samples_split= 2, max_depth= 100, value_type= 'mean'):
        self.root= None
        self.min_samples_split= min_samples_split
        self.max_depth= max_depth
        self.value_type= value_type

    
    def build_tree(self, data, curr_depth= 0):
        X, y= data[:, :-1], data[:, -1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and curr_depth < self.max_depth:
            best_split= self.best_split(data, n_features)
            if best_split['var_reduction'] > 0:
                left= self.build_tree(best_split['left_dataset'], curr_depth + 1)
                right= self.build_tree(best_split['right_dataset'], curr_depth + 1)
                return Node(
                    feature_idx= best_split['feature_idx'], 
                    threshold= best_split['threshold'],
                    var_reduction= best_split['var_reduction'],
                    left= left,
                    right= right
                    )
        
        if self.value_type == 'median':
            return Node(value= np.median(y))
        return Node(value= np.mean(y))



    def best_split(self, data, n_features):
        optimal_split= {'feature_idx': None, 'threshold': None, 'var_reduction': -1, 'left_dataset': None, 'right_dataset': None}

        for feature_idx in range(n_features):
            unique_vals= np.sort(np.unique(data[:, feature_idx]))
            thresholds= (unique_vals[:-1] + unique_vals[1:]) / 2
            for threshold in thresholds:
                left_data, right_data = self._split(data, feature_idx, threshold)
                
                if len(left_data) > 0 and len(right_data) > 0: 
                    left_y, right_y, parent_y= left_data[:, -1], right_data[:, -1], data[:, -1]
                    var_reduction= self.variance_reduction(left_y, right_y, parent_y)

                    if var_reduction > optimal_split['var_reduction']:
                        optimal_split['feature_idx']= feature_idx
                        optimal_split['threshold']= threshold
                        optimal_split['var_reduction']= var_reduction
                        optimal_split['left_dataset']= left_data
                        optimal_split['right_dataset']= right_data
        
        return optimal_split


                
    def _split(self, data, feature_idx, threshold):
        left_data= data[data[:, feature_idx] <= threshold]
        right_data= data[data[:, feature_idx] > threshold]
        return left_data, right_data


    def _variance(self, y):
        y_mean= np.mean(y)
        return np.mean((y- y_mean)**2)


    def variance_reduction(self, left_y, right_y, parent_y):
        left_weight= len(left_y)/ len(parent_y)
        right_weight= len(right_y)/ len(parent_y)

        return self._variance(parent_y) - (left_weight * self._variance(left_y) + right_weight * self._variance(right_y))
    
    
    def fit(self, X, y):
        data= np.concatenate([X, y.reshape(-1, 1)], axis= 1)
        self.root= self.build_tree(data)


    def predict(self, X):
        y_pred= [self._predict_value(row, self.root) for row in X]
        return y_pred
    

    def _predict_value(self, row, node: Node):
        if node.is_leaf_node():
            return node.value
        
        feature_idx, threshold = node.feature_idx, node.threshold   
        if row[feature_idx] <= threshold:
            return self._predict_value(row, node.left)
        else:
            return self._predict_value(row, node.right) 



if __name__ == "__main__":
    arr= np.array([[1, 2, 8],
                   [3, 4, 9],
                   [5, 6, 7]])
    y= arr[:, -1]
    unique_vals= np.unique(y)
    print(np.sort(y))
    print((unique_vals[1:] + unique_vals[:-1])/2)