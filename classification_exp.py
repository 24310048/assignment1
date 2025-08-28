import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV



def classification():
    np.random.seed(42)

    # Code given in the question
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

    # For plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.title('Generated Classification Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()

    # Convert to DataFrame/Series for our implementation
    X_df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    y_series = pd.Series(y, dtype='category')

    # Q2 a) Split data 70-30 and show performance
    train_size = int(0.7 * len(X))
    X_train, X_test = X_df[:train_size], X_df[train_size:]
    y_train, y_test = y_series[:train_size], y_series[train_size:]

    print("=== Q2a) Decision Tree Performance ===")
    for criterion in ["information_gain", "gini_index"]:
        print(f"\nCriterion: {criterion}")
        
        # Train our decision tree
        tree = DecisionTree(criterion=criterion, max_depth=5)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        
        # Calculate metrics
        acc = accuracy(y_pred, y_test)
        print(f"Accuracy: {acc:.4f}")
        
        # Per-class precision and recall
        for cls in y_test.unique():
            prec = precision(y_pred, y_test, cls)
            rec = recall(y_pred, y_test, cls)
            print(f"Class {cls} - Precision: {prec:.4f}, Recall: {rec:.4f}")

    # Q2 b) 5-fold cross-validation with nested CV for optimal depth
    print("\n=== Q2b) Cross-Validation and Optimal Depth ===")

    def cross_validate_depth(X, y, depths, criterion, cv_folds=5):
        """Perform cross-validation for different depths"""
        scores = []
        
        for depth in depths:
            fold_scores = []
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                tree = DecisionTree(criterion=criterion, max_depth=depth)
                tree.fit(X_train_fold, y_train_fold)
                y_pred_fold = tree.predict(X_val_fold)
                
                acc = accuracy(y_pred_fold, y_val_fold)
                fold_scores.append(acc)
            
            avg_score = np.mean(fold_scores)
            scores.append(avg_score)
            print(f"Depth {depth}: Average CV Accuracy = {avg_score:.4f} ± {np.std(fold_scores):.4f}")
        
        return scores

    depths = [2, 3, 4, 5, 6, 7, 8]
    for criterion in ["information_gain", "gini_index"]:
        print(f"\nCriterion: {criterion}")
        scores = cross_validate_depth(X_df, y_series, depths, criterion)
        
        best_depth = depths[np.argmax(scores)]
        print(f"Optimal depth: {best_depth}")
        
        # Plot results
        plt.figure(figsize=(8, 6))
        plt.plot(depths, scores, 'o-')
        plt.title(f'Cross-Validation Accuracy vs Tree Depth ({criterion})')
        plt.xlabel('Max Depth')
        plt.ylabel('CV Accuracy')
        plt.grid(True)
        plt.show()