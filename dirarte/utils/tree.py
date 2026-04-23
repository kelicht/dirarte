import numpy as np
import numba
import json
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
import torch
import torch.nn as nn

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except:
    HAS_LGB = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False



@numba.njit("float64[:](float64[:, :], int64[:], float64[:], int64[:], int64[:], float64[:], boolean)", parallel=True, cache=True)
def _predict_tree(X, feature, threshold, children_left, children_right, value, is_xgboost):
    """ Predict using a single decision tree. """

    y = np.zeros(X.shape[0], dtype=np.float64)
    J = np.zeros(X.shape[0], dtype=np.int64)
    for i in numba.prange(X.shape[0]):
        while feature[J[i]] >= 0:
            if is_xgboost:
                is_branch = (X[i, feature[J[i]]] < threshold[J[i]])
            else:
                is_branch = (X[i, feature[J[i]]] <= threshold[J[i]])
            if is_branch:
                J[i] = children_left[J[i]]
            else:
                J[i] = children_right[J[i]]
        y[i] = value[J[i]]
    return y



class GenericTree():
    """ 
    A generic tree structure to store decision tree parameters from different libraries in a unified format.
    
    Parameters:
    ------------
    node_count : int
        Number of nodes in the tree.
    feature : np.ndarray
        Feature indices for each node.
    threshold : np.ndarray
        Threshold values for each node.
    value : np.ndarray
        Output values for each node.
    children_left : np.ndarray
        Indices of left child nodes.
    children_right : np.ndarray
        Indices of right child nodes.
    is_xgboost : bool
        Whether the tree is from XGBoost (which uses < for branching) or not (which uses <= for branching).
    """
    
    def __init__(
        self,
        node_count: int,
        feature: np.ndarray,
        threshold: np.ndarray,
        value: np.ndarray,
        children_left: np.ndarray,
        children_right: np.ndarray,
        is_xgboost: bool = False,
    ):
        self.node_count = node_count
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.children_left = children_left
        self.children_right = children_right
        self.is_xgboost = is_xgboost
    
    
    def predict(self, X: np.ndarray):
        """ 
        Predict using the decision tree.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
            
        Returns
        -------
        y : np.ndarray
            Predicted values.
        """
        
        y = _predict_tree(X, self.feature, self.threshold, self.children_left, self.children_right, self.value, self.is_xgboost)
        return y
        
    
    def to_string(self):
        """ 
        Convert the tree structure to a string representation for debugging.
        
        Returns
        -------
        tree_str : str
            String representation of the tree structure.
        """
        
        tree_str = ""
        node = 0
        depth = 0
        stack = [(node, depth)]
        while len(stack) > 0:
            node, depth = stack.pop()
            indent = "  " * depth
            if self.feature[node] >= 0:
                if self.is_xgboost:
                    tree_str += f"{indent}Node {node}: feature[{self.feature[node]}] < {self.threshold[node]}\n"
                else:
                    tree_str += f"{indent}Node {node}: feature[{self.feature[node]}] <= {self.threshold[node]}\n"
                stack.append((self.children_right[node], depth + 1))
                stack.append((self.children_left[node], depth + 1))
            else:
                tree_str += f"{indent}Leaf {node}: value = {self.value[node]}\n"
        return tree_str      
    
    
    def to_soft_tree(self, n_features: int, sigma: float = 1.0):
        """
        Convert the generic tree into a soft decision tree implemented in PyTorch.
        
        Parameters
        ----------
        n_features : int
            Number of input features.
        sigma : float, optional (default=1.0)
            Temperature parameter for the sigmoid function in the soft tree.
        
        Returns
        -------
        soft_tree : SoftTree
            A soft decision tree that can be used for gradient-based optimization.
        """
        
        return SoftTree(self, n_features=n_features, sigma=sigma)



class SoftTree(nn.Module):
    """
    A soft decision tree implemented in PyTorch.
    
    Parameters
    ----------
    tree : GenericTree
        The generic tree to be converted into a soft tree.
    n_features : int
        Number of input features.
    sigma : float, optional (default=1.0)
        Temperature parameter for the sigmoid function.
    """
    
    def __init__(
        self,
        tree: GenericTree,
        n_features: int,
        sigma: float = 1.0,
    ):
        super(SoftTree, self).__init__()
        self.tree = tree
        self.sigma = sigma
        self.n_features = n_features

        # Convert the tree parameters into PyTorch tensors and register them as parameters of the module
        self.node_count = tree.node_count
        self.threshold = nn.Parameter(torch.from_numpy(tree.threshold).float())
        self.value = nn.Parameter(torch.from_numpy(tree.value).float())
        self.children_left = tree.children_left
        self.children_right = tree.children_right        
        
        # Create a binary feature selection matrix based on the feature indices of the tree nodes
        feature_indices = tree.feature
        feature_matrix = np.zeros((self.node_count, self.n_features))
        for i in range(self.node_count):
            if feature_indices[i] != -2:
                feature_matrix[i, feature_indices[i]] = 1.0
        self.feature_select = nn.Parameter(torch.from_numpy(feature_matrix).float())


    def forward(self, x: torch.Tensor):
        """
        Forward pass of the soft decision tree.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape (batch_size, 1).
        """        
        
        # Compute the input to each node by multiplying the input features with the feature selection matrix
        node_inputs = torch.matmul(x, self.feature_select.t())
        left_probs = torch.sigmoid(self.sigma * (self.threshold - node_inputs))

        # Compute the probability of reaching each node by traversing the tree from the root to the leaves 
        batch_size = x.size(0)
        mu = [None] * self.node_count
        mu[0] = torch.ones(batch_size, device=x.device)
        for i in range(self.node_count):
            if self.children_left[i] != -1:
                left = self.children_left[i]
                right = self.children_right[i]
                mu[left] = mu[i] * left_probs[:, i]
                mu[right] = mu[i] * (1 - left_probs[:, i])

        # Stack the probabilities of reaching each node into a tensor
        mu_tensors = [m if m is not None else torch.zeros(batch_size, device=x.device) for m in mu]
        mu_stacked = torch.stack(mu_tensors, dim=1)

        # Compute the output of the soft tree as a weighted sum of the leaf values
        is_leaf = torch.tensor(self.children_left == -1, device=x.device).float()
        out = torch.matmul(mu_stacked * is_leaf, self.value)
        
        return out



def _parse_sklearn_trees(estimator):
    """ Parse trees from sklearn ensemble models """
    
    trees = []
    n_estimators = len(estimator.estimators_)
    
    for tree in estimator.estimators_:
    
        # If base estimators are classification trees, compute the predicted probabilities
        if isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier)):
            tree_ = tree.tree_
            value_target = tree_.value[:, 0, 1]
            value_sum = tree_.value[:, 0, :].sum(axis=1)
            value = value_target / value_sum
            
        # If base estimators are regression trees, scale the output values by the number of estimators
        elif isinstance(estimator, (GradientBoostingClassifier)):
            tree_ = tree[0].tree_
            value = tree_.value[:, 0, 0] * n_estimators
            
        # Create GenericTree
        generic_tree = GenericTree(
            node_count=tree_.node_count,
            feature=tree_.feature,
            threshold=tree_.threshold,
            value=value,
            children_left=tree_.children_left,
            children_right=tree_.children_right,
        )
        trees.append(generic_tree)

    return trees


def _parse_lgb_trees(estimator):
    """ Parse trees from LightGBM models """
    
    trees = []
    forest_info = estimator.booster_.dump_model()['tree_info']
    n_estimators = len(forest_info)
    
    for tree_info in forest_info:

        node_count = 2 * tree_info['num_leaves'] - 1
        feature = -2 * np.ones(node_count, dtype=np.int64)
        threshold = -2 * np.ones(node_count, dtype=np.float64)
        value = np.zeros(node_count, dtype=np.float64)
        children_left = -1 * np.ones(node_count, dtype=np.int64)
        children_right = -1 * np.ones(node_count, dtype=np.int64)

        # Traverse the tree for extracting the parameters of each node by DFS. 
        stack = [tree_info['tree_structure']]
        j = 0
        while len(stack) > 0:
            node_info = stack.pop()
            node_info['node_index'] = j
            if 'leaf_index' in node_info:
                value[j] = node_info['leaf_value']
            else:
                feature[j] = node_info['split_feature']
                threshold[j] = node_info['threshold']
                value[j] = node_info['internal_value']
                stack.append(node_info['right_child'])
                stack.append(node_info['left_child'])
            j = j + 1

        # Traverse the tree for extracting its structure. 
        stack = [tree_info['tree_structure']]
        j = 0
        while len(stack) > 0:
            node_info = stack.pop()
            if 'leaf_index' not in node_info:
                children_left[j] = node_info['left_child']['node_index']
                children_right[j] = node_info['right_child']['node_index']
                stack.append(node_info['right_child'])
                stack.append(node_info['left_child'])
            j = j + 1
            
        # Create GenericTree
        generic_tree = GenericTree(
            node_count=node_count,
            feature=feature,
            threshold=threshold,
            value=value * n_estimators,
            children_left=children_left,
            children_right=children_right,
        )
        trees.append(generic_tree)
        
    return trees


def _parse_xgb_trees(estimator):
    """ Parse trees from XGBoost models """
    
    trees = []
    forest_info = [json.loads(tree_info) for tree_info in estimator.get_booster().get_dump(dump_format='json')]
    n_estimators = len(forest_info)
    node_counts = [2 * tree_info.count('leaf') - 1 for tree_info in estimator.get_booster().get_dump(dump_format='json')]

    for (tree_info, node_count) in zip(forest_info, node_counts):
        
        feature = -2 * np.ones(node_count, dtype=np.int64)
        threshold = -2 * np.ones(node_count, dtype=np.float64)
        value = np.zeros(node_count, dtype=np.float64)
        children_left = -1 * np.ones(node_count, dtype=np.int64)
        children_right = -1 * np.ones(node_count, dtype=np.int64)

        # Traverse the tree for extracting the parameters of each node by BFS. 
        queue = [tree_info]
        while len(queue) > 0:
            node_info = queue.pop(0)
            j = node_info['nodeid']
            if 'leaf' in node_info:
                value[j] = node_info['leaf']
            else:
                feature[j] = int(node_info['split'].replace('f', ''))
                threshold[j] = node_info['split_condition']
                children_left[j] = node_info['yes']
                children_right[j] = node_info['no']
                children = node_info['children']
                queue.append(children[0])
                queue.append(children[1])
                
        # Create GenericTree
        generic_tree = GenericTree(
            node_count=node_count,
            feature=feature,
            threshold=threshold,
            value=value * n_estimators,
            children_left=children_left,
            children_right=children_right,
            is_xgboost=True,
        )
        trees.append(generic_tree)
        
    return trees


def parse_trees(estimator):
    """ 
    Parse trees from different ensemble models. 
    
    Parameters
    ----------
    estimator : any
        A trained ensemble model (e.g., RandomForest, XGBoost, LightGBM).
    
    Returns
    -------
    trees : list of GenericTree
        A list of parsed trees in a unified format.
    """
    
    if isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)):
        return _parse_sklearn_trees(estimator)
    elif HAS_LGB and isinstance(estimator, LGBMClassifier):
        return _parse_lgb_trees(estimator)
    elif HAS_XGB and isinstance(estimator, XGBClassifier):
        return _parse_xgb_trees(estimator)
    else:
        raise ValueError("Unsupported estimator type for parsing trees.")


def is_boosted_trees(estimator):
    """ 
    Check if the estimator is a boosted tree model. 
    
    Parameters
    ----------
    estimator : any
        A trained ensemble model (e.g., RandomForest, XGBoost, LightGBM).
        
    Returns
    -------
    bool
        True if the estimator is a boosted tree model, False otherwise.
    """
    
    if isinstance(estimator, GradientBoostingClassifier):
        return True
    elif HAS_LGB and isinstance(estimator, LGBMClassifier):
        return True
    elif HAS_XGB and isinstance(estimator, XGBClassifier):
        return True
    else:
        return False
    

def _get_lgb_intercept(estimator):
    """ Get the intercept term for LightGBM models. """

    # TODO: verify that this is correct!
    base_score = estimator.booster_.dump_model()['average_output']
    intercept = np.log(base_score / (1 - base_score))
    return intercept


def _get_xgb_intercept(estimator):
    """ Get the intercept term for XGBoost models. """

    booster = estimator.get_booster()
    config = json.loads(booster.save_config())
    base_score = json.loads(config['learner']['learner_model_param']['base_score'])
    intercept = np.log(base_score / (1 - base_score))
    return intercept
    

def get_intercept(estimator):
    """ Get the intercept term for the given estimator. """
    
    if HAS_LGB and isinstance(estimator, LGBMClassifier):
        return _get_lgb_intercept(estimator)
    elif HAS_XGB and isinstance(estimator, XGBClassifier):
        return _get_xgb_intercept(estimator)
    else:
        return -0.5