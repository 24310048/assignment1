import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from tree.base import DecisionTree
import tsfel

# Load the processed data
def load_har_data(path="HAR/har_dataset.npz"):
    """Load HAR data from saved NumPy archive"""
    data = np.load(path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

def har_decision_trees(X_train, X_test, y_train, y_test):
    """Train decision trees with different feature sets"""
    
    print("=== HAR Decision Tree Analysis ===")
    
    # Method 1: Raw accelerometer data
    print("\n1. Raw Accelerometer Data")
    X_train_raw = X_train.reshape(X_train.shape[0], -1)  # Flatten
    X_test_raw = X_test.reshape(X_test.shape[0], -1)
    
    dt_raw = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_raw.fit(X_train_raw, y_train)
    y_pred_raw = dt_raw.predict(X_test_raw)
    
    print_metrics("Raw Data", y_test, y_pred_raw)
    
    # Method 2: TSFEL features
    print("\n2. TSFEL Features")
    cfg = tsfel.get_features_by_domain()
    
    X_train_tsfel = extract_tsfel_features(X_train, cfg)
    X_test_tsfel = extract_tsfel_features(X_test, cfg)
    
    dt_tsfel = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_tsfel.fit(X_train_tsfel, y_train)
    y_pred_tsfel = dt_tsfel.predict(X_test_tsfel)
    
    print_metrics("TSFEL Features", y_test, y_pred_tsfel)
    
    # Method 3: Statistical features (simulating dataset features)
    print("\n3. Statistical Features")
    X_train_stats = extract_statistical_features(X_train)
    X_test_stats = extract_statistical_features(X_test)
    
    dt_stats = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_stats.fit(X_train_stats, y_train)
    y_pred_stats = dt_stats.predict(X_test_stats)
    
    print_metrics("Statistical Features", y_test, y_pred_stats)
    
    return {
        'raw': (dt_raw, y_pred_raw, X_train_raw, X_test_raw),
        'tsfel': (dt_tsfel, y_pred_tsfel, X_train_tsfel, X_test_tsfel),
        'stats': (dt_stats, y_pred_stats, X_train_stats, X_test_stats)
    }

def extract_tsfel_features(X, cfg):
    """Extract TSFEL features from accelerometer data"""
    features = []
    for i in range(len(X)):
        sample_features = []
        for axis in range(3):
            axis_features = tsfel.time_series_features_extractor(cfg, X[i, :, axis])
            sample_features.extend(axis_features.values[0])
        features.append(sample_features)
    
    features = np.array(features)
    return np.nan_to_num(features)

def extract_statistical_features(X):
    """Extract basic statistical features"""
    features = []
    for sample in X:
        sample_features = []
        for axis in range(3):
            axis_data = sample[:, axis]
            # Basic statistical features
            sample_features.extend([
                np.mean(axis_data),
                np.std(axis_data),
                np.min(axis_data),
                np.max(axis_data),
                np.median(axis_data),
                np.percentile(axis_data, 25),
                np.percentile(axis_data, 75),
                # Energy and entropy features
                np.sum(axis_data**2),
                -np.sum(axis_data * np.log(axis_data + 1e-10))
            ])
        features.append(sample_features)
    
    return np.array(features)

def print_metrics(method_name, y_true, y_pred):
    """Print classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    print(f"\n{method_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    activity_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                     'SITTING', 'STANDING', 'LAYING']
    
    for i, activity in enumerate(activity_names):
        print(f"{activity}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activity_names, yticklabels=activity_names)
    plt.title(f'Confusion Matrix - {method_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def depth_analysis(X_train, X_test, y_train, y_test, methods_data):
    """Analyze performance vs tree depth"""
    depths = range(2, 9)
    
    results = {}
    for method in ['raw', 'tsfel', 'stats']:
        results[method] = []
        _, _, X_tr, X_te = methods_data[method]
        
        for depth in depths:
            dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt.fit(X_tr, y_train)
            y_pred = dt.predict(X_te)
            accuracy = accuracy_score(y_test, y_pred)
            results[method].append(accuracy)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for method in ['raw', 'tsfel', 'stats']:
        plt.plot(depths, results[method], 'o-', label=method.upper())
    
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Performance vs Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results

def analyze_problem_cases(X_test, y_test, y_pred, subject_ids=None):
    """Analyze where the model performs poorly"""
    
    # Find misclassified samples
    misclassified = y_test != y_pred
    
    print(f"\nMisclassification Analysis:")
    print(f"Total misclassified samples: {misclassified.sum()}")
    print(f"Misclassification rate: {misclassified.sum()/len(y_test):.4f}")
    
    # Analyze by activity
    activity_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                     'SITTING', 'STANDING', 'LAYING']
    
    for i, activity in enumerate(activity_names):
        activity_mask = y_test == (i + 1)
        activity_errors = misclassified[activity_mask].sum()
        activity_total = activity_mask.sum()
        
        if activity_total > 0:
            error_rate = activity_errors / activity_total
            print(f"{activity}: {activity_errors}/{activity_total} ({error_rate:.4f})")
    
    # Most confused pairs
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nMost confused activity pairs:")
    for i in range(len(activity_names)):
        for j in range(len(activity_names)):
            if i != j and cm[i, j] > 0:
                confusion_rate = cm[i, j] / cm[i, :].sum()
                if confusion_rate > 0.1:  # Show confusions > 10%
                    print(f"{activity_names[i]} -> {activity_names[j]}: "
                          f"{cm[i, j]} samples ({confusion_rate:.3f})")

# Main execution
def har_decision_tree_analysis():
    # Load your HAR data
    X_train, X_test, y_train, y_test = load_har_data()
    
    print("Run this with your HAR data loaded!")
    
    # Uncomment when data is available:
    methods_data = har_decision_trees(X_train, X_test, y_train, y_test)
    depth_results = depth_analysis(X_train, X_test, y_train, y_test, methods_data)
    analyze_problem_cases(X_test, y_test, methods_data['tsfel'][1])