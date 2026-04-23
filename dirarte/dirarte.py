import numpy as np

from .basic import FeatureTweakingExplainer
from .utils import Y_TARGET



class DirarteExplainer(FeatureTweakingExplainer):

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
        distribution: str = 'wasserstein',
        epsilon: float | list = 0.05,
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
        self.distribution = distribution
        self.epsilon = epsilon


    def _get_validity(self, X, A):
        """ Evaluate distributionally-robust validity based on the model's prediction probabilities. """

        # Generate counterfactual instances based on the actions
        X_cf = np.repeat(X, A.shape[1], axis=0) + np.concatenate(A, axis=0)
        
        if isinstance(self.epsilon, float):
            epsilon = np.array([self.epsilon])
        else:
            epsilon = np.array(self.epsilon)
        is_valid = np.zeros((epsilon.shape[0], A.shape[0], A.shape[1]), dtype=np.bool_)

        # If using Wasserstein distribution
        if self.distribution == 'wasserstein':
            
            # Compute the prediction probabilities for the counterfactual instances and convert to log-odds
            if self.ensemble_type_ == 'boosting':
                y_prob = self.estimator.predict_proba(X_cf)[:, Y_TARGET]
            else:
                y_1 = np.array([tree.predict(X_cf) for tree in self.trees_]).T.mean(axis=1)
                y_prob = y_1 if Y_TARGET == 1 else 1 - y_1
            margin = np.log(y_prob / (1 - y_prob))
            
            # Compute the margin by subtracting the threshold epsilon from the predicted probabilities
            for i in range(epsilon.shape[0]):
                is_valid[i] = (margin > epsilon[i]).reshape(A.shape[0], A.shape[1])
                        
        # If using chi-squared distribution
        elif self.distribution == 'chi-squared':
            
            # Compute the mean and standard deviation of the predictions made by each tree
            sign = 1 if Y_TARGET == 1 else -1
            y_pred = np.array([sign * tree.predict(X_cf) for tree in self.trees_]).T
            y_mean = y_pred.mean(axis=1)
            y_std = y_pred.std(axis=1)
            
            # Compute the margin by subtracting the scaled standard deviation from the mean prediction
            margin = (y_mean + self.intercept_) / y_std
            for i in range(epsilon.shape[0]):
                is_valid[i] = (margin > np.sqrt(epsilon[i])).reshape(A.shape[0], A.shape[1])

        # If using another distribution, raise an error 
        else:
            raise ValueError('Unsupported distribution type: {}'.format(self.distribution))

        return is_valid



