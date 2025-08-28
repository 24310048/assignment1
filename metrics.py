from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"
    
    return (y_hat == y).sum() / len(y)

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"
    
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_positive = ((y_hat == cls) & (y != cls)).sum()
    
    if true_positive + false_positive == 0:
        return 0.0
    
    return true_positive / (true_positive + false_positive)

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"
    
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_negative = ((y_hat != cls) & (y == cls)).sum()
    
    if true_positive + false_negative == 0:
        return 0.0
    
    return true_positive / (true_positive + false_negative)

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"
    
    return np.sqrt(((y_hat - y) ** 2).mean())

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    assert y_hat.size > 0, "Input series cannot be empty"
    
    return (abs(y_hat - y)).mean()