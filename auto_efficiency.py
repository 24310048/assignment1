import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def auto_efficiency():
    np.random.seed(42)

    # Reading the data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    data = pd.read_csv(url, delim_whitespace=True, header=None,
                    names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                            "acceleration", "model year", "origin", "car name"])

    print("=== Automotive Efficiency Dataset Analysis ===")
    print("Original data shape:", data.shape)
    print("\nFirst few rows:")
    print(data.head())

    # Clean the data
    print("\n=== Data Cleaning ===")

    # Remove car name column (not useful for prediction)
    data = data.drop('car name', axis=1)

    # Handle missing values (marked as '?')
    print("Missing values before cleaning:")
    print(data.isnull().sum())

    # Replace '?' with NaN and convert to numeric
    data = data.replace('?', np.nan)
    for col in data.columns:
        if col != 'car name':
            data[col] = pd.to_numeric(data[col], errors='coerce')

    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    # Fill missing values with median
    data = data.fillna(data.median())

    print("Data shape after cleaning:", data.shape)

    # Prepare features and target
    X = data.drop('mpg', axis=1)
    y = data['mpg']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Train our implementation
    print("\n=== Our Decision Tree Implementation ===")
    our_tree = DecisionTree(criterion="information_gain", max_depth=6)
    our_tree.fit(X_train, y_train)
    our_pred = our_tree.predict(X_test)

    our_rmse = rmse(our_pred, y_test)
    our_mae = mae(our_pred, y_test)

    print(f"Our Implementation - RMSE: {our_rmse:.4f}")
    print(f"Our Implementation - MAE: {our_mae:.4f}")

    # Train sklearn implementation
    print("\n=== Sklearn Decision Tree Comparison ===")
    sklearn_tree = DecisionTreeRegressor(max_depth=6, random_state=42)
    sklearn_tree.fit(X_train, y_train)
    sklearn_pred = sklearn_tree.predict(X_test)

    sklearn_rmse = np.sqrt(np.mean((sklearn_pred - y_test)**2))
    sklearn_mae = np.mean(np.abs(sklearn_pred - y_test))

    print(f"Sklearn Implementation - RMSE: {sklearn_rmse:.4f}")
    print(f"Sklearn Implementation - MAE: {sklearn_mae:.4f}")

    # Performance comparison
    print(f"\n=== Performance Comparison ===")
    print(f"RMSE Difference: {abs(our_rmse - sklearn_rmse):.4f}")
    print(f"MAE Difference: {abs(our_mae - sklearn_mae):.4f}")

    # Visualize predictions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, our_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual MPG')
    plt.ylabel('Predicted MPG')
    plt.title(f'Our Implementation\nRMSE: {our_rmse:.3f}')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, sklearn_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual MPG')
    plt.ylabel('Predicted MPG')
    plt.title(f'Sklearn Implementation\nRMSE: {sklearn_rmse:.3f}')

    plt.tight_layout()
    plt.show()