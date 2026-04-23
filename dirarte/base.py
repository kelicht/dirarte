import numpy as np
from sklearn.ensemble import IsolationForest
from abc import ABC, abstractmethod

from .utils import (
    parse_trees,
    get_intercept,
    is_boosted_trees,
    compute_percentiles,
    compute_weights,
)



class Recourse():
    """
    A class to represent the recourse explanation for an instance.

    Parameters
    ------------
    action : np.ndarray of shape (n_samples, n_features)
        The recommended actions for recourse, where each row corresponds to an instance and each column corresponds to a feature. 
    counterfactual : np.ndarray of shape (n_samples, n_features)
        The counterfactual instances after applying the recommended actions, where each row corresponds to an instance and each column corresponds to a feature.
    cost : np.ndarray of shape (n_samples,)
        The cost of the recommended actions for each instance, where each value indicates the cost of the corresponding action.
    probability : np.ndarray of shape (n_samples,)
        The predicted probability of the target class for the counterfactual instances, where each value indicates the predicted probability for the corresponding counterfactual instance.
    validity : np.ndarray of shape (n_samples,)
        Boolean values indicating whether the counterfactual instances are predicted as the target class by the model, where each value is True if the corresponding counterfactual instance is predicted as the target class and False otherwise.
    plausibility : np.ndarray of shape (n_samples,)
        The plausibility scores of the counterfactual instances, where each value indicates the plausibility of the corresponding counterfactual instance based on outlier scores. Higher values indicate less plausible counterfactuals.
    """
    
    def __init__(
        self, 
        action: np.ndarray, 
        counterfactual: np.ndarray, 
        cost: np.ndarray, 
        probability: np.ndarray, 
        validity: np.ndarray, 
        plausibility: np.ndarray
    ):
        
        self.action = action
        self.counterfactual = counterfactual
        self.cost = cost
        self.probability = probability
        self.validity = validity
        self.plausibility = plausibility


    def get_average_cost(self):
        """ 
        Evaluate average cost of valid recourse actions. 
        The cost is averaged only over the valid counterfactual instances, which are those that are predicted as the target class by the model. 
        
        Returns
        ------------
        average_cost : float
            The average cost of valid recourse actions.
        """
        
        average_cost = self.cost[self.validity].mean()
        return average_cost
        
    def get_average_plausibility(self):
        """ 
        Evaluate average plausibility of valid counterfactuals. 
        The plausibility is averaged only over the valid counterfactual instances, which are those that are predicted as the target class by the model. Higher values indicate less plausible counterfactuals.
        
        Returns
        ------------
        average_plausibility : float
            The average plausibility of valid counterfactuals.
        """
        
        average_plausibility = self.plausibility[self.validity].mean()
        return average_plausibility

    def get_average_sparsity(self):
        """ 
        Evaluate average sparsity of valid counterfactuals. 
        The sparsity is calculated as the average number of features that are changed in the valid counterfactual instances, where a feature is considered changed if the corresponding action value is non-zero.
        
        Returns
        ------------
        average_sparsity : float
            The average sparsity of valid counterfactuals.
        """

        average_sparsity = (self.action[self.validity] != 0).mean(axis=0).sum()
        return average_sparsity
    


class BaseExplainer(ABC):
    """
    An abstract base class for recourse explainers. This class defines the interface for generating recourse explanations and can be extended to implement specific explainer methods.
    
    Parameters
    ------------
    estimator : any
        A trained tree ensemble model (e.g., RandomForest, XGBoost, LightGBM).
    constraints : dict
        A dictionary containing feature constraints, including:
            - is_binary: np.ndarray of bool, indicating binary features;
            - is_integer: np.ndarray of bool, indicating integer features;
            - is_immutable: np.ndarray of bool, indicating immutable features;
            - is_unincreasable: np.ndarray of bool, indicating features that cannot be increased;
            - is_irreducible: np.ndarray of bool, indicating features that cannot be reduced;
            - categories: list of list of int, indicating groups of categorical feature indices.
    cost_type : str, optional
        Type of cost metric ('mps', 'tlps', 'mad', 'std', 'normalized', 'uniform') (default is 'tlps').
    cost_ord : int, optional
        Order of the norm-based cost metric (1 for L1, 2 for L2, -1 for L-infinity) (default is 1).
    """
        
    def __init__(
        self,
        estimator: any, 
        constraints: dict,
        cost_type: str = 'tlps',
        cost_ord: int = 1,
    ):
        
        self.estimator = estimator
        self.constraints = constraints
        self.cost_type = cost_type
        self.cost_ord = cost_ord


    def initialize(
        self, 
        X: np.ndarray
    ):
        """ 
        Initialize the explainer by parsing the trees, computing regions, and preparing for generating actions. 
        This method should be called before generating recourse explanations. 
        
        Parameters
        ------------
        X : np.ndarray of shape (n_samples, n_features)
            The input data used for initializing the explainer, which is typically the training data. 
            This data is used to compute feature ranges, percentiles, weights, and regions for generating actions.
            
        Returns
        ------------
        self : RecourseExplainer
            The initialized explainer object.
        """
        
        self.n_features_ = X.shape[1]
        self.feature_range_ = np.array([
            (X[:, d].min(), X[:, d].max()) for d in range(self.n_features_)
        ])
        
        # Compute weights or percentiles based on cost type
        if self.cost_type in ['mps', 'tlps']:
            is_immutable = self.constraints['is_immutable']
            self.weights_ = None
            self.percentiles_ = compute_percentiles(X, is_immutable)
        else:
            is_binary = self.constraints['is_binary']
            self.weights_ = compute_weights(X, self.cost_type, is_binary)
            self.percentiles_ = None

        # Initialize Isolation Forest for evaluating plausibility of counterfactuals
        self.iforest_ = IsolationForest().fit(X)
        
        # Parse trees in the ensemble model
        self.trees_ = parse_trees(self.estimator)        
        self.intercept_ = get_intercept(self.estimator)
        self.ensemble_type_ = 'boosting' if is_boosted_trees(self.estimator) else 'bagging'
        
        return self
    
    
    @abstractmethod
    def explain_recourse(
        self, 
        X: np.ndarray, 
    ) -> Recourse | list:
        """
        Generate recourse explanations for the given instances X, which are predicted as the undesired class by the model. 
        
        Parameters
        ------------
        X : np.ndarray of shape (n_samples, n_features)
            The input instances for which to generate recourse explanations. 
            These instances should be predicted as the undesired class by the model.
            
        Returns
        ------------
        recourse : Recourse | list of Recourse
            An object containing the recourse explanations, including recommended actions, 
            counterfactual instances, costs, predicted probabilities, validity, and plausibility.
        """
        
        pass


