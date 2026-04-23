import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))



def process_dataset(
    df: pd.DataFrame, 
    target_column: str, 
    target_name: str,
    immutable_columns: list = [],
    unincreasable_columns: list = [],
    irreducible_columns: list = [],
    prefix_sep: str = ':'
):  
    """
    Process dataset for recourse analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_column : str
        Name of the target column.
    target_name : str
        Name of the target class.
    immutable_columns : list of str, optional
        List of immutable feature names.
    unincreasable_columns : list of str, optional
        List of unincreasable feature names.
    irreducible_columns : list of str, optional
        List of irreducible feature names.
    prefix_sep : str, optional
        Prefix separator for categorical features.
        
    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    constraints : dict
        Dictionary of feature constraints.
    """
    
    # Process target variable and features
    y = (df[target_column].values == target_name).astype(np.int64)
    df_processed = pd.get_dummies(df.drop(target_column, axis=1), prefix_sep=prefix_sep)
    X = df_processed.values.astype(np.float64)   
    
    # Generate feature constraints
    constraints = _get_constraints(
        df_processed, 
        immutable_columns,
        unincreasable_columns,
        irreducible_columns,
        prefix_sep
    )
    
    return X, y, constraints


def _get_constraints(df_processed, immutable_columns, unincreasable_columns, irreducible_columns, prefix_sep):
    """ Generate feature constraints for recourse. """ 
    
    feature_names = df_processed.columns.values
    n_features = len(feature_names)

    # Determine feature types (binary, integer, continuous)
    is_binary = np.zeros(n_features, dtype=bool)
    is_integer = np.zeros(n_features, dtype=bool)
    for d, feature in enumerate(feature_names):
        if df_processed[feature].dtype == float:
            continue
        elif np.array_equal(np.array([0, 1]), np.sort(df_processed[feature].unique())):
            is_binary[d] = True
            is_integer[d] = True
        else:
            is_integer[d] = True

    # Determine feature constraints (immutable, unincreasable, irreducible)
    is_immutable = np.zeros(n_features, dtype=bool)
    is_unincreasable = np.zeros(n_features, dtype=bool)
    is_irreducible = np.zeros(n_features, dtype=bool)
    for d, feature in enumerate(feature_names):
        if prefix_sep in feature:
            feature, _ = feature.split(prefix_sep)
        if feature in immutable_columns:
            is_immutable[d] = True
        elif feature in unincreasable_columns:
            is_unincreasable[d] = True
        elif feature in irreducible_columns:
            is_irreducible[d] = True
    
    # Determine categorical feature groups
    categories = []
    prefix = ''
    _categories = []
    for d, feature in enumerate(feature_names):
        if prefix_sep not in feature:
            continue
        prefix_d, _ = feature.split(prefix_sep)
        if prefix == prefix_d:
            _categories.append(d)
        else:
            if len(_categories) > 0:
                categories.append(_categories)
            prefix = prefix_d
            _categories = [d]
    if len(_categories) > 0:
        categories.append(_categories)
        
    constraints = {
        'feature_names': feature_names, 
        'is_binary': is_binary,
        'is_integer': is_integer,
        'is_immutable': is_immutable,
        'is_unincreasable': is_unincreasable,
        'is_irreducible': is_irreducible,
        'categories': categories
    }
    return constraints



class Dataset():
    """
    Dataset helper class for processing and retrieving datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_column : str
        Name of the target column.
    target_name : str
        Name of the target class. 
    immutable_columns : list of str, optional
        List of immutable feature names.
    unincreasable_columns : list of str, optional
        List of unincreasable feature names.
    irreducible_columns : list of str, optional
        List of irreducible feature names.
        
    Attributes
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    constraints : dict
        Dictionary of feature constraints.
    name : str
        Name of the dataset.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        target_name: str,
        immutable_columns: list = [],
        unincreasable_columns: list = [],
        irreducible_columns: list = [],  
        name: str = None      
    ):

        self.X, self.y, self.constraints = process_dataset(
            df, 
            target_column, 
            target_name,
            immutable_columns,
            unincreasable_columns,
            irreducible_columns
        )
        self.name = name


    def get_dataset(self, split=False, test_size=0.2, validation_size=0.0):
        """
        Retrieve the dataset, optionally splitting into training, validation, and test sets.

        Parameters
        ----------
        split : bool, optional
            Whether to split the dataset (default is False).
        test_size : float, optional
            Proportion of the dataset to include in the test split (default is 0.2).
        validation_size : float, optional
            Proportion of the dataset to include in the validation split (default is 0.0).
        """

        if split:
            X_tr, X_ts, y_tr, y_ts = train_test_split(self.X, self.y, test_size=test_size+validation_size, stratify=self.y)
            if validation_size > 0:
                X_vl, X_ts, y_vl, y_ts = train_test_split(X_ts, y_ts, test_size=test_size/(test_size+validation_size), stratify=y_ts)
                return X_tr, X_vl, X_ts, y_tr, y_vl, y_ts
            else:
                return X_tr, X_ts, y_tr, y_ts
        else:
            return self.X, self.y   


    def get_shifted_dataset(self, test_size=0.2, shift_type='split', delete_fraction=0.1, label_shift_fraction=0.1):
        """
        Retrieve shifted dataset for evaluating recourse robustness.
        
        Parameters
        ----------
        test_size : float, optional
            Proportion of the dataset to include in the test split (default is 0.2).
        shift_type : str, optional
            Type of distribution shift to simulate ('split', 'delete', 'label') (default is 'split').
        delete_fraction : float, optional
            Fraction of samples to delete, available for 'delete' shift type (default is 0.1).
        label_shift_fraction : float, optional
            Fraction of samples to shift labels, available for 'label' shift type (default is 0.1).
        """
        
        X_train, X_test, y_train, y_test = self.get_dataset(split=True, test_size=test_size)
        n_samples, n_features = X_train.shape

        # Split the dataset into two halves.
        if shift_type == 'split':
            X_before, X_after, y_before, y_after = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train)

        # Randomly delete a proportion of samples from the dataset.
        elif shift_type == 'delete':
            n_delete = int(delete_fraction * n_samples)
            is_remaining = np.random.choice(n_samples, n_samples - n_delete, replace=False)
            X_before, X_after, y_before, y_after = X_train, X_train[is_remaining], y_train, y_train[is_remaining]
                        
        # Split the dataset into two halves with different label distributions.
        elif shift_type == 'label':
            n_half = n_samples // 2
            base_positive_ratio = (y_train == 1).mean()
            ratio_before = base_positive_ratio * (1 + label_shift_fraction)
            ratio_after = base_positive_ratio * (1 - label_shift_fraction)
            n_positive_before, n_positive_after =int(n_half * ratio_before), int(n_half * ratio_after)
            n_negative_before, n_negative_after = n_half - n_positive_before, n_half - n_positive_after
            is_positive, is_negative = np.where(y_train == 1)[0], np.where(y_train == 0)[0]
            is_before = np.concatenate([np.random.choice(is_positive, n_positive_before, replace=False), np.random.choice(is_negative, n_negative_before, replace=False)])
            is_after = np.concatenate([np.random.choice(is_positive, n_positive_after, replace=False), np.random.choice(is_negative, n_negative_after, replace=False)])
            X_before, X_after, y_before, y_after = X_train[is_before], X_train[is_after], y_train[is_before], y_train[is_after]

        return X_before, X_after, X_test, y_before, y_after, y_test


    def get_details(self):
        """ 
        Retrieve dataset feature details as a DataFrame. 
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing feature details, including name, type, min, max, immutability, and constraints.
        """
        
        features = self.constraints['feature_names']
        n_features = len(features)
        types = ['Binary' if self.constraints['is_binary'][d] else ('Integer' if self.constraints['is_integer'][d] else 'Real') for d in range(n_features)]
        mins, maxs = self.X.min(axis=0), self.X.max(axis=0)
        immutables = ['Yes' if self.constraints['is_immutable'][d] else 'No' for d in range(n_features)]
        constraints = ['Fix' if self.constraints['is_immutable'][d] else ('Unincreasable' if self.constraints['is_unincreasable'][d] else ('Irreducible' if self.constraints['is_irreducible'][d] else 'Nothing')) for d in range(n_features)]
        details = {
            'Feature': features,
            'Type': types, 
            'Min': mins,
            'Max': maxs, 
            'Immutable': immutables, 
            'Constraint': constraints,
        }
        return pd.DataFrame(details)   
    


class FicoDataset(Dataset):
    """ FICO dataset helper class. """
    
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/fico.csv')
        immutable_columns = [
            'NumTrades60Ever2DerogPubRec',
            'NumTrades90Ever2DerogPubRec',
            'MaxDelq2PublicRecLast12M',
            'MaxDelqEver'
        ]
        super().__init__(
            df=df,
            target_column='RiskPerformance',
            target_name='Good',
            immutable_columns=immutable_columns, 
            name='FICO'
        )



class AdultDataset(Dataset):
    """ Adult dataset helper class. """
    
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/adult.csv')
        immutable_columns = ['marital-status', 'race', 'gender', 'native-country']
        irreducible_columns = ['age']
        super().__init__(
            df=df,
            target_column='income',
            target_name='>50K',
            immutable_columns=immutable_columns,
            irreducible_columns=irreducible_columns,
            name='Adult'
        )

        

class CreditDataset(Dataset):
    """ Credit dataset helper class. """
    
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/credit.csv')
        immutable_columns = ['Married', 'Single']
        irreducible_columns = ['Age']
        super().__init__(
            df=df,
            target_column='DefaultNextMonth',
            target_name='No',
            immutable_columns=immutable_columns,
            irreducible_columns=irreducible_columns,
            name='Credit'
        )
      
        
            
class GermanDataset(Dataset):
    """ German dataset helper class. """
    
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/german.csv')
        immutable_columns = ['Sex', 'Housing']
        irreducible_columns = ['Age']
        super().__init__(
            df=df,
            target_column='GoodCustomer',
            target_name='Yes',
            immutable_columns=immutable_columns,
            irreducible_columns=irreducible_columns,
            name='German'
        )



class SbaDataset(Dataset):
    """ Small Business Administration (SBA) dataset helper class. """
    
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/sba.csv')
        immutable_columns = ['NAICS', 'NewExist', 'UrbanRural']
        super().__init__(
            df=df,
            target_column='Default',
            target_name='No',
            immutable_columns=immutable_columns,
            name='SBA'
        )



class LoanDataset(Dataset):
    """ Loan Prediction dataset helper class. """
    
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/loan.csv')
        immutable_columns = ['Gender', 'Married', 'SelfEmployed', 'CreditHistory']        
        super().__init__(
            df=df,
            target_column='LoanStatus',
            target_name='Y',
            immutable_columns=immutable_columns,
            name='Loan'
        )



class CompasDataset(Dataset):
    """ COMPAS dataset helper class. """
    
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/compas.csv')
        immutable_columns = ['sex', 'race']
        irreducible_columns = ['age']
        super().__init__(
            df=df,
            target_column='two_year_recid',
            target_name='No',
            immutable_columns=immutable_columns,
            irreducible_columns=irreducible_columns,
            name='COMPAS'
        )
        


class BailDataset(Dataset):
    """ Bail dataset helper class. """
    
    def __init__(self):
        df = pd.read_csv(CURRENT_DIR + '/bail.csv')
        immutable_columns = ['White', 'Married', 'Male']
        super().__init__(
            df=df,
            target_column='Recidivate',
            target_name='No',
            immutable_columns=immutable_columns,
            name='Bail'
        )
