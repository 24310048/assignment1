import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    encoded_dfs = []
    
    for col in X.columns:
        if X[col].dtype == 'category' or X[col].dtype == 'object':
            # One-hot encode categorical columns
            encoded = pd.get_dummies(X[col], prefix=col, drop_first=False)
            encoded_dfs.append(encoded)
        else:
            # Keep numerical columns as is
            encoded_dfs.append(X[[col]])
    
    return pd.concat(encoded_dfs, axis=1)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    # Check if it's categorical or object type
    if y.dtype == 'category' or y.dtype == 'object':
        return False
    
    # Check if all values are integers (but dtype might be float)
    if y.dtype in ['int64', 'int32', 'int16', 'int8']:
        return False
        
    # For float types, check if they're actually discrete
    if len(y.unique()) < 10 and all(val.is_integer() for val in y.dropna() if isinstance(val, (int, float))):
        return False
        
    return True

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    if len(Y) == 0:
        return 0
    
    value_counts = Y.value_counts()
    probabilities = value_counts / len(Y)
    
    # Calculate entropy: -sum(p * log2(p))
    entropy_val = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy_val

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    if len(Y) == 0:
        return 0
    
    value_counts = Y.value_counts()
    probabilities = value_counts / len(Y)
    
    # Calculate Gini index: 1 - sum(p^2)
    gini = 1 - sum(p**2 for p in probabilities)
    return gini

def variance(Y: pd.Series) -> float:
    """
    Function to calculate variance for regression
    """
    if len(Y) == 0:
        return 0
    return Y.var()

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if len(Y) == 0:
        return 0
    
    # Calculate initial impurity
    if check_ifreal(Y):
        initial_impurity = variance(Y)
    else:
        if criterion == "gini_index":
            initial_impurity = gini_index(Y)
        else:  # information_gain (entropy)
            initial_impurity = entropy(Y)
    
    # Calculate weighted impurity after split
    weighted_impurity = 0
    unique_values = attr.unique()
    
    for value in unique_values:
        mask = (attr == value)
        subset_y = Y[mask]
        weight = len(subset_y) / len(Y)
        
        if check_ifreal(Y):
            subset_impurity = variance(subset_y)
        else:
            if criterion == "gini_index":
                subset_impurity = gini_index(subset_y)
            else:  # information_gain (entropy)
                subset_impurity = entropy(subset_y)
        
        weighted_impurity += weight * subset_impurity
    
    return initial_impurity - weighted_impurity

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    """
    best_gain = -1
    best_attribute = None
    best_threshold = None
    
    for feature in features:
        if X[feature].dtype in ['category', 'object'] or len(X[feature].unique()) <= 10:
            # Discrete feature
            gain = information_gain(y, X[feature], criterion)
            if gain > best_gain:
                best_gain = gain
                best_attribute = feature
                best_threshold = None
        else:
            # Continuous feature - try different thresholds
            unique_values = sorted(X[feature].unique())
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                binary_attr = X[feature] <= threshold
                gain = information_gain(y, binary_attr, criterion)
                
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = feature
                    best_threshold = threshold
    
    return best_attribute, best_threshold

def split_data(X: pd.DataFrame, y: pd.Series, attribute, threshold=None):
    """
    Function to split the data according to an attribute.
    """
    if threshold is None:
        # Discrete attribute
        unique_values = X[attribute].unique()
        splits = {}
        for value in unique_values:
            mask = X[attribute] == value
            splits[value] = (X[mask], y[mask])
        return splits
    else:
        # Continuous attribute
        left_mask = X[attribute] <= threshold
        right_mask = ~left_mask
        
        return {
            'left': (X[left_mask], y[left_mask]),
            'right': (X[right_mask], y[right_mask])
        }