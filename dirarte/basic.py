import numpy as np
from scipy.special import expit
from pyscipopt import Model
import torch

from .base import Recourse, BaseExplainer
from .utils import (
    MIN_VAL,
    MAX_VAL,
    Y_TARGET,
    prune_trees,
    compute_regions,
    merge_regions,
    compute_all_actions,
    is_feasible,
    find_best_actions,
    flatten,
    LinSum,
    LinExpr,
    compute_leaf_indicators,
)



class FeatureTweakingExplainer(BaseExplainer):
    """
    Generate recourse explanations for tree ensemble models by the extended feature tweaking algorithm.
    
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
    max_features : int, optional
        Maximum number of features that can be changed (default is -1, meaning no limit).
    plausibility : float, optional
        Plausibility threshold for filtering counterfactuals based on their outlier scores (default is 1.0).
    prune_level : float, optional
        Level of pruning trees in the ensemble as a fraction of total trees (default is 1.0, meaning no pruning).
    merge_level : int, optional
        Level of merging regions from different trees (default is 2).
    max_search : float, optional
        Maximum number of actions to consider for recourse as a fraction of total regions (default is 1.0, meaning no limit).
    temperature : float, optional
        Noise strength for Gumbel-Top-k sampling when max_search < 1.0 (default is 0.0, meaning no noise).
    tolerance : float, optional
        Tolerance to adjust feature steps when generating actions (default is 0.05).
    """
    
    def __init__(
        self, 
        estimator: any, 
        constraints: dict,
        cost_type: str = 'tlps',
        cost_ord: int = 1,
        max_features: int = -1,
        plausibility: float = 1.0,
        prune_level: float = 1.0,
        merge_level: int = 2,
        max_search: float = 1.0,
        temperature: float = 0.0,
        tolerance: float = 0.05,
    ):
        
        super().__init__(
            estimator, 
            constraints,
            cost_type, 
            cost_ord, 
        )        
        self.max_features = max_features
        self.plausibility = plausibility
        self.prune_level = prune_level
        self.merge_level = merge_level     
        self.max_search = max_search   
        self.temperature = temperature
        self.tolerance = tolerance
    
    
    def initialize(self, X: np.ndarray):
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
        
        # Prepare cost functions, fit isolation forest, and parse trees in the ensemble model
        self = super().initialize(X)
        
        # Prune trees in the ensemble to reduce the number of regions if prune_level < 1.0
        if self.prune_level < 1.0:
            n_select = int(self.estimator.n_estimators * (1 - self.prune_level))
            is_remained = prune_trees(self.estimator, X, n_select)
            trees = [self.trees_[t] for t in is_remained]
        else:
            trees = self.trees_      
        
        # Compute regions for all leaf nodes in all trees and merge them
        self.regions_ = self._get_regions(trees)
        
        return self


    def _get_regions(self, trees):
        """ Compute regions for all leaf nodes in the passed decision trees. """
        
        # Compute regions for all leaf nodes in all trees
        regions = []
        values = []
        for tree in trees:
            tree_regions = compute_regions(
                tree.feature,
                tree.threshold,
                tree.children_left,
                tree.children_right,
                self.n_features_,
                tree.node_count,
            )       
            leaf_regions = tree_regions[tree.feature < 0]
            leaf_values = tree.value[tree.feature < 0]
            regions.append(leaf_regions)
            values.append(leaf_values)
        
        # Merge regions from different trees by computing their intersections
        for _ in range(self.merge_level - 1):
            n_regions = len(regions)
            if n_regions < 2:
                break
            merged_regions = []
            merged_values = []
            for t in range(int(n_regions / 2)):
                regions1 = regions[2 * t]
                regions2 = regions[2 * t + 1]
                values1 = values[2 * t]
                values2 = values[2 * t + 1]
                regions_and_values = merge_regions(regions1, regions2, values1, values2)
                new_regions = regions_and_values[:, :-1, :]
                new_values = regions_and_values[:, -1].mean(axis=1)
                merged_regions.append(new_regions)
                merged_values.append(new_values)
            regions = merged_regions
            values = merged_values
        regions = np.concatenate(regions, axis=0)
        values = np.concatenate(values, axis=0)
        
        # Randomly sample via Gumbel-Top-k trick
        if self.max_search < 1.0:
            n_search = int(regions.shape[0] * self.max_search)
            sign = 1 if Y_TARGET == 1 else -1
            prob = expit(sign * values)
            noise = -np.log(-np.log(np.random.uniform(size=prob.shape[0])))
            idx = np.argpartition(prob + self.temperature * noise, -n_search)[-n_search:]
            regions = regions[idx]            

        # Null region covering the entire feature space
        null_region = np.zeros((1, self.n_features_, 2), dtype=np.float64)
        null_region[0, :, 0] = MIN_VAL
        null_region[0, :, 1] = MAX_VAL
        regions = np.concatenate([null_region, regions], axis=0)
        
        return regions


    def explain_recourse(self, X: np.ndarray):
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
        
        # Generate possible actions for recourse
        A = self._get_actions(X)

        # Evaluate feasibility and cost of actions
        feasibility = self._get_feasibility(X, A)
        cost = self._get_costs(X, A)

        # Find the best action for each instance
        best = find_best_actions(X, A, feasibility, cost)
        n_thresholds = best.shape[0]

        recourses = []
        for t in range(n_thresholds):

            # Compile results
            A_best, cost_best = best[t, :, 1:], best[t, :, 0]
            counterfactuals = X + A_best
            probability = self.estimator.predict_proba(counterfactuals)[:, Y_TARGET]
            validity = (self.estimator.predict(counterfactuals) == Y_TARGET)
            plausibility = -1 * self.iforest_.score_samples(counterfactuals)

            # Create Recourse object for the current threshold
            recourse = Recourse(
                action=A_best,
                counterfactual=counterfactuals,
                cost=cost_best,
                probability=probability,
                validity=validity,
                plausibility=plausibility
            )       
            recourses.append(recourse) 

        if n_thresholds == 1:
            return recourses[0]
        else:
            return recourses
    

    def _get_actions(self, X):
        """ Generate all possible actions for recourse. """

        # Compute all possible actions
        is_binary = self.constraints['is_binary']
        is_integer = self.constraints['is_integer']
        feature_steps = []
        for d in range(self.n_features_):
            if is_binary[d]:
                feature_steps.append(0.0)
            else:
                feature_step = (self.feature_range_[d][1] - self.feature_range_[d][0]) * self.tolerance
                if is_integer[d]:
                    feature_step = np.floor(feature_step)
                feature_steps.append(feature_step)
        feature_steps = np.array(feature_steps, dtype=np.float64)
        A = compute_all_actions(X, self.regions_, feature_steps, is_binary, is_integer)
        
        # Adjust categorical constraints
        categories = self.constraints['categories']
        if len(categories) > 0:
            cost_category = np.zeros(self.n_features_, dtype=np.float64)
            for cats in categories:
                if self.cost_type == 'mps':
                    cost_category[cats] += np.array([
                        MAX_VAL if self.percentiles_[d] is None 
                        else abs(self.percentiles_[d](1) - self.percentiles_[d](0)) for d in cats])                
                elif self.cost_type == 'tlps':
                    cost_category[cats] += np.array([
                        MAX_VAL if self.percentiles_[d] is None 
                        else abs(np.log2((1-self.percentiles_[d](1)) / (1-self.percentiles_[d](0)))) for d in cats])                
                else:
                    cost_category[cats] += self.weights_[cats]        

            def adjust(x, A_x):
                for cats in categories:
                    i = A_x[:, cats].sum(axis=1)
                    if (i == 1).sum() > 0:
                        d = cats[np.where(x[cats] == 1)[0][0]]
                        A_x[i == 1, d] = -1.0
                    if (i == -1).sum() > 0:
                        for l in np.arange(self.regions_.shape[0])[i == -1]:
                            r = self.regions_[l]
                            cats_feasible = [cats[j] for j, r_d in enumerate(r[cats]) if r_d[1] == MAX_VAL]
                            if len(cats_feasible) == 0:
                                A_x[l, cats] = 0.0
                            else:
                                idx = np.argmin(cost_category[cats_feasible])
                                d_feasible = cats_feasible[idx]
                                A_x[l, d_feasible] = 1.0
                return A_x              

            A = np.array([adjust(X[i], A[i]) for i in range(X.shape[0])])
            
        return A
    
    
    def _get_feasibility(self, X, A):
        """ Check feasibility of actions based on constraints. """

        # Check feasibility of actions based on constraints
        is_immutable = self.constraints['is_immutable']
        is_unincreasable = self.constraints['is_unincreasable']
        is_irreducible = self.constraints['is_irreducible']
        F = is_feasible(A, is_immutable, is_unincreasable, is_irreducible, self.max_features)
        
        # Check feasibility of actions based on categorical constraints
        categories = self.constraints['categories']
        if len(categories) > 0:
            for cats in categories:
                F = F * (A[:, :, cats].sum(axis=2) == 0)

        # Check plausibility of actions based on outlier scores
        P = self._get_plausibility(X, A)
        F = F * (P <= self.plausibility)

        # Check validity of actions based on model predictions
        V = self._get_validity(X, A)
        F = F * V
        
        return F            
    
    
    def _get_validity(self, X, A):
        """ Check validity of actions based on model predictions. """
        
        # Compute predictions for all counterfactual instances generated by applying actions
        X_cf = np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0)
        if self.ensemble_type_ == 'boosting':
            y_pred = self.estimator.predict_proba(X_cf)[:, Y_TARGET]
        else:
            y_1 = np.array([tree.predict(X_cf) for tree in self.trees_]).T.mean(axis=1)
            y_pred = y_1 if Y_TARGET == 1 else 1 - y_1
        
        # Check validity of actions based on whether the predicted class is the target class
        is_valid = np.zeros((1, A.shape[0], A.shape[1]), dtype=np.bool_)
        is_valid[0] = (y_pred > 0.5).reshape(A.shape[0], A.shape[1])

        return is_valid
        

    def _get_costs(self, X, A):
        """ Evaluate costs of actions for recourse. """
        
        def cost(x, A_x):
            if self.cost_type in ['mps', 'tlps']:
                C = np.zeros(A_x.shape[0])
                for d in range(A_x.shape[1]):
                    q_d = self.percentiles_[d]
                    if q_d is None:
                        continue
                    if self.cost_type == 'mps':
                        C = np.maximum(C, abs(q_d(x[d]) - q_d(x[d] + A_x[:, d])))
                    else:
                        C += abs( np.log2( (1 - q_d(x[d] + A_x[:, d])) / (1 - q_d(x[d])) ) )
            else:
                if self.cost_ord == -1:
                    C = np.max(abs(A_x) * self.weights_, axis=1)
                else:
                    C = (abs(A_x) ** self.cost_ord).dot(self.weights_) 
            return C

        return np.array([cost(X[i], A[i]) for i in range(X.shape[0])])    
    

    def _get_plausibility(self, X, A):
        """ Evaluate plausibility of actions based on outlier scores. """

        # Generate counterfactual instances based on the actions
        X_cf = np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0)

        # Compute outlier scores using Isolation Forest and determine plausibility based on the threshold
        plausibility = -1 * self.iforest_.score_samples(X_cf).reshape(A.shape[0], A.shape[1])
        
        return plausibility



class OptimalActionExplainer(BaseExplainer):
    """
    Generate recourse explanations for tree ensemble models by the optimal action extraction algorithm.
    
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
    max_features : int, optional
        Maximum number of features that can be changed (default is -1, meaning no limit).
    time_limit : int, optional
        Time limit for generating recourse explanations in seconds (default is 60).
    hide_output : bool, optional
        Whether to hide output during the explanation process (default is True).
    """
    
    def __init__(
        self, 
        estimator: any, 
        constraints: dict,
        cost_type: str = 'tlps',
        cost_ord: int = 1,
        max_features: int = -1,
        time_limit: int = 60,
        hide_output: bool = True,
    ):
        
        super().__init__(
            estimator, 
            constraints,
            cost_type, 
            cost_ord, 
        )        
        self.max_features = max_features
        self.time_limit = time_limit
        self.hide_output = hide_output


    def initialize(self, X: np.ndarray):
        
        # Prepare cost functions, fit isolation forest, and parse trees in the ensemble model
        self = super().initialize(X)
        
        # Get thresholds for each feature from all trees
        all_thresholds = dict([(f, []) for f in range(self.n_features_)])
        for tree in self.trees_:
            features = tree.feature
            thresholds = tree.threshold
            for f, t in zip(features, thresholds):
                if f >= 0:
                    all_thresholds[f].append(t)
        self.thresholds_ = {f: np.unique(t) for f, t in all_thresholds.items()}    
        
        # Get regions for all leaf nodes in all trees
        regions = []
        values = []
        for tree in self.trees_:
            tree_regions = compute_regions(
                tree.feature,
                tree.threshold,
                tree.children_left,
                tree.children_right,
                self.n_features_,
                tree.node_count,
            )       
            leaf_regions = tree_regions[tree.feature < 0]
            leaf_values = tree.value[tree.feature < 0]
            regions.append(leaf_regions)
            values.append(leaf_values)
        self.regions_ = regions
        self.values_ = values

        return self
    

    def explain_recourse(self, X: np.ndarray):
        
        n_samples = X.shape[0]
        A_best = np.zeros_like(X)
        cost_best = np.zeros(n_samples)
        
        # Compute the optimal action for each instance by solving a mixed-integer linear program
        for n in range(n_samples):
            a_n, c_n = self._compute_optimal_action(X[n])
            A_best[n] = a_n
            cost_best[n] = c_n
            
        # Compile results
        counterfactuals = X + A_best
        probability = self.estimator.predict_proba(counterfactuals)[:, Y_TARGET]
        validity = (self.estimator.predict(counterfactuals) == Y_TARGET)
        plausibility = -1 * self.iforest_.score_samples(counterfactuals)

        recourse = Recourse(
            action=A_best,
            counterfactual=counterfactuals,
            cost=cost_best,
            probability=probability,
            validity=validity,
            plausibility=plausibility
        )        
        return recourse
            

    def _compute_optimal_action(self, x):
        
        # Generate feasible counterfactuals and compute cost coefficients
        feasible_counterfactuals = self._get_feasible_counterfactuals(x)
        cost_coefficients = self._get_cost_coefficients(x, feasible_counterfactuals)
        n_cfs = np.array([len(cfs) for cfs in feasible_counterfactuals])
        cf_ptr = np.concatenate(([0], np.cumsum(n_cfs)))        
        
        # Compute leaf indicators for each feasible counterfactual based on the regions of all leaf nodes in all trees
        cf_values = np.concatenate(feasible_counterfactuals, axis=0).astype(np.float64)        
        regions = np.concatenate(self.regions_, axis=0).astype(np.float64)
        is_xgboost = self.trees_[0].is_xgboost
        is_cf_in_leaf = compute_leaf_indicators(cf_values, regions, is_xgboost)
        
        # Define decision variables
        model = Model()
        n_estimators = len(self.trees_)
        n_leaves = np.array([values.shape[0] for values in self.values_])        
        x_cf = [model.addVar(vtype='C', lb=self.feature_range_[d, 0], ub=self.feature_range_[d, 1], name='cf_{:04d}'.format(d)) for d in range(self.n_features_)]
        cost = model.addVar(vtype='C', lb=0, ub=MAX_VAL, name='cost')
        phi = [[model.addVar(vtype='B', name='phi_{:04d}_{:04d}'.format(t, l)) for l in range(n_leaves[t])] for t in range(n_estimators)]
        pi = [[model.addVar(vtype='B', name='pi_{:04d}_{:04d}'.format(d, m)) for m in range(n_cfs[d])] for d in range(self.n_features_)]
        
        # Set objective function and constraints
        model = self._set_objective_function(model, cost, pi, cost_coefficients)
        model = self._set_basic_constraints(model, pi, x_cf, cf_values, cf_ptr, n_cfs)
        model = self._set_prediction_constraints(model, phi, n_estimators)
        model = self._set_decision_logic_constraints(model, phi, pi, is_cf_in_leaf, n_estimators, n_leaves)
                
        # Solve the optimization problem
        model.hideOutput(self.hide_output)
        model.setParam('limits/time', self.time_limit)
        model.optimize()    

        # Extract the optimal action and cost from the solution
        if model.getStatus() == 'infeasible':
            a_opt = np.zeros(self.n_features_)
            c_opt = 0.0
        else:
            x_cf = np.array([model.getVal(x_cf[d]) for d in range(self.n_features_)])
            x_cf = np.clip(x_cf, self.feature_range_[:, 0], self.feature_range_[:, 1])
            a_opt = x_cf - x
            c_opt = model.getVal(cost)        

        return a_opt, c_opt
        

    def _get_feasible_counterfactuals(self, x):
        """ Generate feasible counterfactual values for each feature based on the constraints and thresholds from the trees. """
        
        feasible_counterfactuals = []
        for d in range(self.n_features_):

            # If the feature is immutable or has no thresholds, the only feasible counterfactual value is the original value
            if self.constraints['is_immutable'][d] or (self.thresholds_[d].shape[0] == 0):
                X_d = np.array([x[d]])

            # If the feature is binary, the feasible counterfactual values are the original value and its complement (1 - original value)
            elif self.constraints['is_binary'][d]:
                X_d = np.array([x[d], 1 - x[d]])

            # Otherwise, the feasible counterfactual values are the original value and the unique thresholds from the trees
            else:
                X_d = self.thresholds_[d].copy()
                
                # If the feature is integer, adjust the thresholds to be the nearest integers above or below the original value
                if self.constraints['is_integer'][d]:
                    X_d[X_d >= x[d]] = np.floor(X_d[X_d >= x[d]]) + 1
                    X_d[X_d < x[d]] = np.floor(X_d[X_d < x[d]])

                # If the feature is continuous, adjust the thresholds to be slightly above or below the original value
                else:
                    X_d[X_d >= x[d]] = X_d[X_d >= x[d]] + 1e-8
                    X_d[X_d < x[d]] = X_d[X_d < x[d]] - 1e-8
                    
                # Remove duplicate values and ensure the original value is included in the feasible counterfactuals
                X_d = np.unique(X_d)
                X_d = X_d[X_d != x[d]]
                X_d = np.concatenate(([x[d]], X_d))

            # If there are constraints on the direction of change for the feature, filter the feasible counterfactual values accordingly
            if self.constraints['is_unincreasable'][d]:
                X_d = X_d[X_d <= x[d]]
            elif self.constraints['is_irreducible'][d]:
                X_d = X_d[X_d >= x[d]]

            # Append the feasible counterfactual values with their corresponding feature inde
            feasible_counterfactuals.append(np.array([(d, v) for v in X_d]))
            
        return feasible_counterfactuals
    

    def _get_cost_coefficients(self, x, feasible_counterfactuals):
        """ Compute cost coefficients for each feasible counterfactual value based on the specified cost function. """
        
        cost_coeffs = []
        for d in range(self.n_features_):            
            cf_d = feasible_counterfactuals[d][:, 1]

            # If there is only one feasible counterfactual value (i.e., the original value), the cost coefficient is zero
            if len(cf_d) == 1:
                cost_coeffs.append(np.array([0.0]))

            else:
                
                # If the cost type is MPS or TLPS, compute the cost coefficients based on the percentiles
                if self.cost_type in ['mps', 'tlps']:
                    q_d = self.percentiles_[d]
                    if self.cost_type == 'mps':
                        c_d = abs(q_d(x[d]) - q_d(cf_d))
                    else:
                        c_d = abs( np.log2( (1 - q_d(cf_d)) / (1 - q_d(x[d])) ) )
                    cost_coeffs.append(c_d)

                # Otherwise, compute the cost coefficients based on the weights and the specified norm
                else:
                    absolute_actions = np.abs(cf_d - x[d])
                    if self.cost_ord == -1:
                        cost_coeffs.append(absolute_actions * self.weights_[d])
                    else:
                        absolute_actions = absolute_actions ** self.cost_ord
                        cost_coeffs.append(absolute_actions * self.weights_[d])

        return cost_coeffs
    
    
    def _set_objective_function(self, model, cost, pi, cost_coefficients):
        """ Define the objective function based on the cost function and the cost coefficients for each feasible counterfactual value. """

        # Define the objective function based on the cost function
        model.setObjective(cost, sense='minimize')
        if self.cost_type == 'mps' or self.cost_ord == -1:
            for d in range(self.n_features_):
                model.addCons(cost - LinExpr(cost_coefficients[d], pi[d]) >= 0)
        else:
            cost_coeffs_flatten = np.concatenate(cost_coefficients, axis=0)
            pi_flatten = flatten(pi)
            model.addCons(cost - LinExpr(cost_coeffs_flatten, pi_flatten) == 0)        

        return model        
        
    
    def _set_basic_constraints(self, model, pi, x_cf, cf_values, cf_ptr, n_cfs):
        """ Set basic constraints on feature values and the selection of counterfactual values for each feature. """
        
        # Constraint: only one counterfactual value can be selected for each feature
        for d in range(self.n_features_):
            model.addCons(LinSum(pi[d]) == 1)
            model.addCons(x_cf[d] - LinExpr(cf_values[cf_ptr[d]:cf_ptr[d+1], 1], pi[d]) == 0)
        
        # Constraint: the number of features changed should be less than max_features
        if self.max_features > 0:
            pi_flatten = flatten([pi[d][1:] for d in range(self.n_features_) if n_cfs[d] > 1])
            model.addCons(LinSum(pi_flatten) <= self.max_features)

        # Constraint: categorical features should satisfy the one-hot encoding constraint
        for G in self.constraints['categories']:
            model.addCons(LinSum([x_cf[d] for d in G]) == 1)        
            
        return model
            
    
    def _set_prediction_constraints(self, model, phi, n_estimators):
        """ Set constraints to ensure that the prediction for the counterfactual instance is the target class. """

        # Constraint: the prediction for the counterfactual instance should be the target class
        values_flatten = np.concatenate(self.values_, axis=0)
        phi_flatten = flatten(phi)
        if Y_TARGET == 1:
            model.addCons((1 / n_estimators) * LinExpr(values_flatten, phi_flatten) + self.intercept_ >= 1e-8)
        else:    
            model.addCons((1 / n_estimators) * LinExpr(values_flatten, phi_flatten) + self.intercept_ <= - 1e-8)

        return model
    
    
    def _set_decision_logic_constraints(self, model, phi, pi, is_cf_in_leaf, n_estimators, n_leaves):
        """ Set constraints to ensure that the selected counterfactual instance falls into the leaf nodes corresponding to the selected actions. """

        # Constraint: each leaf node should have at most one counterfactual instance (decision logic constraints)
        leaf_ptr = np.concatenate(([0], np.cumsum(n_leaves)))
        for t in range(n_estimators):
            I_t = is_cf_in_leaf[leaf_ptr[t]:leaf_ptr[t+1]].astype(int)
            model.addCons(LinSum(phi[t]) == 1)
            for l in range(n_leaves[t]):
                model.addCons(self.n_features_ * phi[t][l] - LinExpr(I_t[l], flatten(pi)) <= 0)
        
        return model
    
    

class FocusExplainer(BaseExplainer):
    """
    Generate recourse explanations for tree ensemble models by the FOCUS algorithm.
    
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
        Type of cost metric ('mad', 'std', 'normalized', 'uniform') (default is 'std').
    cost_ord : int, optional
        Order of the norm-based cost metric (1 for L1 or 2 for L2) (default is 1).
    """
    
    def __init__(
        self, 
        estimator: any, 
        constraints: dict,
        cost_type: str = 'std',
        cost_ord: int = 1,
        sigma: float = 1.0,
        beta: float = 0.05,
        max_iter: int = 100,
        lr: float = 0.1,
    ):
        
        super().__init__(
            estimator, 
            constraints,
            cost_type, 
            cost_ord, 
        )        
        self.sigma = sigma
        self.beta = beta
        self.max_iter = max_iter
        self.lr = lr


    def initialize(self, X: np.ndarray):
        
        # Prepare cost functions, fit isolation forest, and parse trees in the ensemble model
        self = super().initialize(X)
        
        # Convert the hard trees into soft trees
        self.soft_trees_ = [tree.to_soft_tree(self.n_features_, self.sigma) for tree in self.trees_]

        # FocusExplainer does not support MPS and TLPS cost types
        if self.cost_type in ['mps', 'tlps']:
            raise ValueError("Cost type '{}' is not supported by FocusExplainer.".format(self.cost_type))
        
        # Convert weights to a PyTorch tensor for efficient computation during optimization
        else:
            self.weights_ = torch.from_numpy(self.weights_).float()
        
        # FocusExplainer only supports L1 and L2 norm-based cost metrics
        if self.cost_ord == -1:
            raise ValueError("Cost order '{}' is not supported by FocusExplainer.".format(self.cost_ord))
        
        return self
    
    
    def explain_recourse(self, X: np.ndarray):
        
        n_samples = X.shape[0]
        A_best = np.zeros_like(X)
        cost_best = np.zeros(n_samples)
        
        # Compute the optimal action for each instance by solving a mixed-integer linear program
        for n in range(n_samples):
            a_n, c_n = self._optimize_action(X[n])
            A_best[n] = a_n
            cost_best[n] = c_n
            
        # Compile results
        counterfactuals = X + A_best
        probability = self.estimator.predict_proba(counterfactuals)[:, Y_TARGET]
        validity = (self.estimator.predict(counterfactuals) == Y_TARGET)
        plausibility = -1 * self.iforest_.score_samples(counterfactuals)

        recourse = Recourse(
            action=A_best,
            counterfactual=counterfactuals,
            cost=cost_best,
            probability=probability,
            validity=validity,
            plausibility=plausibility
        )        
        return recourse
    
    
    def _optimize_action(self, x):

        # Transform the input instance into a PyTorch tensor
        x_tensor = torch.tensor(x.reshape(1, -1)).float()
        x_tensor.requires_grad = True

        # optimize the action by gradient descent
        optimizer = torch.optim.Adam([x_tensor], lr=self.lr)
        for i in range(self.max_iter):
            optimizer.zero_grad()

            # Compute the output of the soft trees for the current counterfactual instance
            output = torch.zeros(len(self.soft_trees_))
            for t, soft_tree in enumerate(self.soft_trees_):
                output[t] = soft_tree(x_tensor)

            # Compute the cost of the action based on the specified cost function
            cost = (self.weights_ * torch.abs(x_tensor - torch.tensor(x).float()) ** self.cost_ord).sum()

            # Compute the loss as a sum of the negative log-probability of the target class and the cost of the action
            if Y_TARGET == 1:
                if self.ensemble_type_ == 'boosting':
                    loss = - torch.log(torch.sigmoid(output.mean() + self.intercept_)) + self.beta * cost
                else:
                    loss = - torch.log(output.mean()) + self.beta * cost
            else:
                if self.ensemble_type_ == 'boosting':
                    loss = - torch.log(1 - torch.sigmoid(output.mean() + self.intercept_)) + self.beta * cost
                else:
                    loss = - torch.log(1 - output.mean()) + self.beta * cost
            
            # Perform backpropagation and update the counterfactual instance
            loss.backward()
            optimizer.step()
            
            # Project the counterfactual instance back to the feasible region defined by the constraints
            with torch.no_grad():
                for d in range(self.n_features_):
                    if self.constraints['is_immutable'][d]:
                        x_tensor[0, d] = x[d]
                    elif self.constraints['is_unincreasable'][d]:
                        x_tensor[0, d] = torch.min(x_tensor[0, d], torch.tensor(x[d]).float())
                    elif self.constraints['is_irreducible'][d]:
                        x_tensor[0, d] = torch.max(x_tensor[0, d], torch.tensor(x[d]).float())

        # Extract the optimized counterfactual instance
        x_cf = x_tensor.detach().numpy()[0]
        
        # Project the optimized counterfactual instance back to the feasible region defined by the constraints
        for d in range(self.n_features_):
            if self.constraints['is_binary'][d]:
                x_cf[d] = 1.0 if x_cf[d] >= 0.5 else 0.0
            elif self.constraints['is_integer'][d]:
                x_cf[d] = np.round(x_cf[d])
            x_cf[d] = np.clip(x_cf[d], self.feature_range_[d][0], self.feature_range_[d][1])
        
        # Compute the action and its cost
        a_opt = x_cf - x
        c_opt = (self.weights_ * np.abs(a_opt) ** self.cost_ord).sum()
        
        return a_opt, c_opt
