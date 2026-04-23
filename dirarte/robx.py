import numpy as np
from sklearn.neighbors import KDTree

from .base import Recourse
from .basic import FeatureTweakingExplainer, OptimalActionExplainer
from .utils import Y_TARGET, find_best_actions



class RobxExplainer(FeatureTweakingExplainer):

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
        tau: float | list = 0.5,
        n_neighbors: int = 1000,
        sigma: float = 0.1,
        n_conservative: int = 10,
        n_interpolate: int = 10,
    ):
        super().__init__(
            estimator, 
            constraints, 
            cost_type, 
            cost_ord, 
            max_features, 
            plausibility,
            prune_level,
            merge_level, 
            max_search, 
            temperature, 
            tolerance
        )   
        self.tau = tau
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.n_conservative = n_conservative
        self.n_interpolate = n_interpolate
        

    def initialize(self, X):
        
        # Initialize parent class
        self = super().initialize(X)

        # Compute standard deviation of features for noise generation in stability evaluation
        self.std_ = X.std(axis=0)
        
        # Build KDTree of stable instances
        X_target = X[self.estimator.predict(X) == Y_TARGET]
        if isinstance(self.tau, float):
            taus = np.array([self.tau])
        else:
            taus = np.array(self.tau)
        self.neighbors_ = {}
        for tau in taus:
            is_stable = (self._evaluate_stability(X_target) >= tau)
            self.neighbors_[tau] = KDTree(X_target[is_stable])
        
        return self


    def _evaluate_stability(self, X):
        
        n_samples, n_features = X.shape
        
        # Generate noisy samples around the input instances
        noise = np.random.normal(
            loc=0.0, 
            scale=self.sigma * self.std_, 
            size=(self.n_neighbors * n_samples, n_features)
        )
        N_x = np.repeat(X, self.n_neighbors, axis=0) + noise
        
        # Compute the prediction probabilities for the noisy samples and evaluate stability
        if self.ensemble_type_ == 'boosting':
            y_proba = self.estimator.predict_proba(N_x)[:, Y_TARGET]
        else:
            y_1 = np.array([tree.predict(N_x) for tree in self.trees_]).T.mean(axis=1)
            y_proba = y_1 if Y_TARGET == 1 else 1 - y_1
        y_proba = y_proba.reshape(n_samples, self.n_neighbors)
        stability = np.mean(y_proba, axis=1) - np.std(y_proba, axis=1)

        return stability    

    
    def explain_recourse(self, X):
        
        # Get initial recourse explanations
        initial_recourse = super().explain_recourse(X)

        # If tau is a single float, convert it to an array for consistent processing in the loop below
        if isinstance(self.tau, float):
            taus = np.array([self.tau])
        else:
            taus = np.array(self.tau)

        # Iterate over each tau threshold to refine counterfactuals to be robust based on stability evaluation
        recourses = []
        for tau in taus:
        
            # Refine counterfactuals for robustness
            X_cf = initial_recourse.counterfactual.copy()
            A = self._generate_interpolated_actions(X, X_cf, tau)

            # Evaluate feasibility and loss of actions
            feasibility = self._get_feasibility(X, A)
            cost = self._get_costs(X, A)
            
            # Evaluate stability of actions and filter based on tau threshold
            X_cf = np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0)
            stability = self._evaluate_stability(X_cf).reshape(A.shape[0], A.shape[1])
            
            # If tau is a single float, convert it to an array for consistent processing 
            is_stable = (stability >= tau).reshape(1, A.shape[0], A.shape[1])
            feasibility = feasibility[0] * is_stable

            # Find the best action for each instance
            best = find_best_actions(X, A, feasibility, cost)

            # Compile results
            A_best, cost_best = best[0, :, 1:], best[0, :, 0]
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

        if len(recourses) == 1:
            return recourses[0]
        else:
            return recourses

    
    def _generate_interpolated_actions(self, X, X_cf, tau):

        n_samples, n_features = X.shape
        n_candidates = self.n_conservative * self.n_interpolate
        
        # Find nearest neighbors for each counterfactual instance and generate conservative directions
        X_neighbors = np.asarray(self.neighbors_[tau].data)
        conservative_indices = self.neighbors_[tau].query(X_cf, k=self.n_conservative, return_distance=False)     
        X_conservative = X_neighbors[conservative_indices]        
        directions = X_conservative - X_cf[:, np.newaxis, :]
        
        # Apply constraints to the directions based on the defined constraints for each feature
        is_immutable = self.constraints['is_immutable']
        is_unincreasable = self.constraints['is_unincreasable']
        is_irreducible = self.constraints['is_irreducible']
        directions[:, :, is_immutable] = 0.0
        directions[:, :, is_unincreasable] = np.minimum(0.0, directions[:, :, is_unincreasable])
        directions[:, :, is_irreducible] = np.maximum(0.0, directions[:, :, is_irreducible])
        
        # Interpolate between the original counterfactual and the conservative directions to generate new candidate actions
        steps = np.arange(1, self.n_interpolate + 1) * (1.0 / self.n_interpolate)
        interpolated_diff = directions[:, :, np.newaxis, :] * steps[np.newaxis, np.newaxis, :, np.newaxis]
        X_cand = X_cf[:, np.newaxis, np.newaxis, :] + interpolated_diff

        # Round binary features to the nearest integer and reshape to get the final candidate actions
        is_binary = self.constraints['is_binary']
        X_cand[:, :, :, is_binary] = np.round(X_cand[:, :, :, is_binary])
        
        # Compute the actions by subtracting the original instances from the candidate counterfactuals
        A = X_cand - X[:, np.newaxis, np.newaxis, :]
        A = A.reshape(n_samples, n_candidates, n_features)

        return A    



