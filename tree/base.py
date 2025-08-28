from dataclasses import dataclass
from typing import Literal, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class Node:
    def __init__(self):
        self.attribute = None
        self.threshold = None
        self.children = {}
        self.value = None
        self.is_leaf = False

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_regression = None

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
        """
        Recursive function to build the decision tree
        """
        node = Node()
        
        # Base cases
        if depth >= self.max_depth or len(y.unique()) == 1 or len(X) == 0:
            node.is_leaf = True
            if self.is_regression:
                node.value = y.mean()
            else:
                node.value = y.mode().iloc[0] if len(y.mode()) > 0 else y.iloc[0]
            return node
        
        # Find best split
        features = X.columns
        best_attr, best_threshold = opt_split_attribute(X, y, self.criterion, features)
        
        if best_attr is None:
            node.is_leaf = True
            if self.is_regression:
                node.value = y.mean()
            else:
                node.value = y.mode().iloc[0] if len(y.mode()) > 0 else y.iloc[0]
            return node
        
        node.attribute = best_attr
        node.threshold = best_threshold
        
        # Split data and create children
        splits = split_data(X, y, best_attr, best_threshold)
        
        for split_value, (X_subset, y_subset) in splits.items():
            if len(X_subset) > 0:
                node.children[split_value] = self._build_tree(X_subset, y_subset, depth + 1)
            else:
                # Create leaf node with parent's majority class/mean
                leaf = Node()
                leaf.is_leaf = True
                if self.is_regression:
                    leaf.value = y.mean()
                else:
                    leaf.value = y.mode().iloc[0] if len(y.mode()) > 0 else y.iloc[0]
                node.children[split_value] = leaf
        
        return node

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # Determine if it's regression or classification
        self.is_regression = check_ifreal(y)
        
        # One-hot encode categorical features
        X_encoded = one_hot_encoding(X)
        
        # Build the tree
        self.root = self._build_tree(X_encoded, y, 0)
        self.feature_names = X_encoded.columns

    def _predict_sample(self, sample: pd.Series) -> Union[float, Any]:
        """
        Predict a single sample
        """
        node = self.root
        
        while not node.is_leaf:
            if node.threshold is None:
                # Discrete attribute
                attr_value = sample[node.attribute]
                if attr_value in node.children:
                    node = node.children[attr_value]
                else:
                    # Use the first available child if exact match not found
                    if node.children:
                        node = list(node.children.values())[0]
                    break
            else:
                # Continuous attribute
                if sample[node.attribute] <= node.threshold:
                    node = node.children.get('left', node)
                else:
                    node = node.children.get('right', node)
                
                if node is None:
                    break
        
        return node.value if node else (0 if self.is_regression else 0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        # One-hot encode with same structure as training
        X_encoded = one_hot_encoding(X)
        
        # Ensure same columns as training data
        for col in self.feature_names:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        X_encoded = X_encoded[self.feature_names]
        
        predictions = []
        for idx, row in X_encoded.iterrows():
            pred = self._predict_sample(row)
            predictions.append(pred)
        
        return pd.Series(predictions, index=X.index)

    def _plot_tree(self, node: Node, depth: int = 0, prefix: str = "") -> None:
        """
        Recursive function to plot the tree
        """
        if node.is_leaf:
            print(f"{prefix}Class/Value: {node.value}")
            return
        
        if node.threshold is None:
            # Discrete attribute
            print(f"{prefix}?({node.attribute})")
            for value, child in node.children.items():
                print(f"{prefix}  {value}:")
                self._plot_tree(child, depth + 1, prefix + "    ")
        else:
            # Continuous attribute
            print(f"{prefix}?({node.attribute} <= {node.threshold:.3f})")
            if 'left' in node.children:
                print(f"{prefix}  Y:")
                self._plot_tree(node.children['left'], depth + 1, prefix + "    ")
            if 'right' in node.children:
                print(f"{prefix}  N:")
                self._plot_tree(node.children['right'], depth + 1, prefix + "    ")

    def plot(self) -> None:
        """
        Function to plot the tree
        """
        if self.root is None:
            print("Tree has not been trained yet!")
            return
        
        print("Decision Tree Structure:")
        print("=" * 50)
        self._plot_tree(self.root)
        print("=" * 50)
    
    