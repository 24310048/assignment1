import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tsfel
import warnings
warnings.filterwarnings('ignore')

# Load the processed data
def load_har_data(path="HAR/har_dataset_full.npz"):
    """Load HAR full dataset (X, y, classes)"""
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    classes = data["classes"].item()  # convert object back to dict
    return X, y, classes


# Task 1: EDA Questions

def plot_activity_waveforms(X, y, classes):
    """Plot waveform for one sample from each activity class"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    activity_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                     'SITTING', 'STANDING', 'LAYING']
    
    for i, activity in enumerate(activity_names):
        # Find first sample of this activity
        activity_idx = classes[activity]
        sample_idx = np.where(y == activity_idx)[0][0]
        sample_data = X[sample_idx]
        
        # Plot all three axes
        time = np.arange(len(sample_data))
        axes[i].plot(time, sample_data[:, 0], label='acc_x', alpha=0.7)
        axes[i].plot(time, sample_data[:, 1], label='acc_y', alpha=0.7)
        axes[i].plot(time, sample_data[:, 2], label='acc_z', alpha=0.7)
        
        axes[i].set_title(f'{activity}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Acceleration')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_static_vs_dynamic(X, y, classes):
    """Analyze linear acceleration for static vs dynamic activities"""
    # Calculate total acceleration
    total_acc = np.sqrt(np.sum(X**2, axis=2))
    
    static_activities = ['SITTING', 'STANDING', 'LAYING']
    dynamic_activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']
    
    static_data = []
    dynamic_data = []
    
    for activity in static_activities:
        activity_idx = classes[activity]
        mask = y == activity_idx
        static_data.extend(total_acc[mask].flatten())
    
    for activity in dynamic_activities:
        activity_idx = classes[activity]
        mask = y == activity_idx
        dynamic_data.extend(total_acc[mask].flatten())
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(static_data, bins=50, alpha=0.7, label='Static Activities', density=True)
    plt.hist(dynamic_data, bins=50, alpha=0.7, label='Dynamic Activities', density=True)
    plt.xlabel('Total Acceleration')
    plt.ylabel('Density')
    plt.title('Distribution of Total Acceleration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([static_data, dynamic_data], labels=['Static', 'Dynamic'])
    plt.ylabel('Total Acceleration')
    plt.title('Boxplot of Total Acceleration')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Static activities mean acceleration: {np.mean(static_data):.4f}")
    print(f"Dynamic activities mean acceleration: {np.mean(dynamic_data):.4f}")
    print(f"Clear separation: {'Yes' if abs(np.mean(static_data) - np.mean(dynamic_data)) > 0.5 else 'No'}")

def pca_visualization(X, y, classes):
    """Visualize data using PCA with different feature extraction methods"""
    
    # Method 1: Total Acceleration
    print("Computing PCA on Total Acceleration...")
    total_acc = np.sqrt(np.sum(X**2, axis=2))
    
    scaler = StandardScaler()
    total_acc_scaled = scaler.fit_transform(total_acc)
    
    pca = PCA(n_components=2)
    total_acc_pca = pca.fit_transform(total_acc_scaled)
    
    # Method 2: TSFEL Features
    print("Computing TSFEL features...")
    cfg = tsfel.get_features_by_domain()
    
    tsfel_features = []
    for i in range(len(X)):
        sample_features = []
        for axis in range(3):
            features = tsfel.time_series_features_extractor(cfg, X[i, :, axis])
            sample_features.extend(features.values[0])
        tsfel_features.append(sample_features)
    
    tsfel_features = np.array(tsfel_features)
    tsfel_features = np.nan_to_num(tsfel_features)  # Handle NaN values
    
    scaler2 = StandardScaler()
    tsfel_scaled = scaler2.fit_transform(tsfel_features)
    
    pca2 = PCA(n_components=2)
    tsfel_pca = pca2.fit_transform(tsfel_scaled)
    
    # Method 3: Flattened raw data (simulating dataset features)
    print("Computing PCA on raw features...")
    raw_features = X.reshape(X.shape[0], -1)  # Flatten
    
    scaler3 = StandardScaler()
    raw_scaled = scaler3.fit_transform(raw_features)
    
    pca3 = PCA(n_components=2)
    raw_pca = pca3.fit_transform(raw_scaled)
    
    # Plot all three methods
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    activity_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                     'SITTING', 'STANDING', 'LAYING']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Plot 1: Total Acceleration PCA
    for i, activity in enumerate(activity_names):
        activity_idx = classes[activity]
        mask = y == activity_idx
        axes[0].scatter(total_acc_pca[mask, 0], total_acc_pca[mask, 1], 
                       c=colors[i], label=activity, alpha=0.6)
    axes[0].set_title('PCA - Total Acceleration')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: TSFEL PCA
    for i, activity in enumerate(activity_names):
        activity_idx = classes[activity]
        mask = y == activity_idx
        axes[1].scatter(tsfel_pca[mask, 0], tsfel_pca[mask, 1], 
                       c=colors[i], label=activity, alpha=0.6)
    axes[1].set_title('PCA - TSFEL Features')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Raw features PCA
    for i, activity in enumerate(activity_names):
        activity_idx = classes[activity]
        mask = y == activity_idx
        axes[2].scatter(raw_pca[mask, 0], raw_pca[mask, 1], 
                       c=colors[i], label=activity, alpha=0.6)
    axes[2].set_title('PCA - Raw Features')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print explained variance
    print(f"Total Acceleration PCA - Explained Variance: {pca.explained_variance_ratio_}")
    print(f"TSFEL PCA - Explained Variance: {pca2.explained_variance_ratio_}")
    print(f"Raw Features PCA - Explained Variance: {pca3.explained_variance_ratio_}")

def correlation_analysis(X, y):
    """Calculate correlation matrix for different feature sets"""
    
    # TSFEL features correlation
    cfg = tsfel.get_features_by_domain()
    tsfel_features = []
    feature_names = []
    
    # Get feature names
    sample_features = tsfel.time_series_features_extractor(cfg, X[0, :, 0])
    feature_names = list(sample_features.columns)
    
    for i in range(len(X)):
        sample_features = []
        for axis in range(3):
            features = tsfel.time_series_features_extractor(cfg, X[i, :, axis])
            sample_features.extend(features.values[0])
        tsfel_features.append(sample_features)
    
    tsfel_features = np.array(tsfel_features)
    tsfel_features = np.nan_to_num(tsfel_features)
    
    # Create feature names for all axes
    all_feature_names = []
    for axis in ['x', 'y', 'z']:
        for name in feature_names:
            all_feature_names.append(f"{name}_{axis}")
    
    # Calculate correlation matrix
    tsfel_df = pd.DataFrame(tsfel_features, columns=all_feature_names)
    correlation_matrix = tsfel_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                vmin=-1, vmax=1, center=0)
    plt.title('TSFEL Features Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                      correlation_matrix.columns[j], corr_val))
    
    print(f"\nHighly correlated feature pairs (|correlation| > 0.8):")
    for pair in high_corr_pairs[:10]:  # Show top 10
        print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

def har_exploratory():
    # Load your data here
    X, y, classes = load_har_data()
    
    classes = {"WALKING": 1, "WALKING_UPSTAIRS": 2, "WALKING_DOWNSTAIRS": 3, 
               "SITTING": 4, "STANDING": 5, "LAYING": 6}
    
    print("=== HAR Exploratory Data Analysis ===")
    
    # Uncomment these when you have data loaded:
    plot_activity_waveforms(X, y, classes)
    analyze_static_vs_dynamic(X, y, classes)
    pca_visualization(X, y, classes)
    correlation_analysis(X, y)