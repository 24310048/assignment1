import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *


def experiments():
    np.random.seed(42)
    num_average_time = 10  # Number of times to run each experiment

    def create_fake_data(N, P, case_type):
        """Create fake data for different cases"""
        if case_type == "discrete_discrete":
            # Discrete input, discrete output
            X = pd.DataFrame({f'feature_{i}': pd.Series(np.random.randint(0, 5, N), dtype='category') 
                            for i in range(P)})
            y = pd.Series(np.random.randint(0, 3, N), dtype='category')
        
        elif case_type == "discrete_real":
            # Discrete input, real output
            X = pd.DataFrame({f'feature_{i}': pd.Series(np.random.randint(0, 5, N), dtype='category') 
                            for i in range(P)})
            y = pd.Series(np.random.randn(N))
        
        elif case_type == "real_discrete":
            # Real input, discrete output
            X = pd.DataFrame(np.random.randn(N, P))
            y = pd.Series(np.random.randint(0, 3, N), dtype='category')
        
        else:  # real_real
            # Real input, real output
            X = pd.DataFrame(np.random.randn(N, P))
            y = pd.Series(np.random.randn(N))
        
        return X, y

    def measure_time(N_values, P_values, case_type):
        """Measure training and prediction time for different N and P values"""
        
        # Vary N (keeping P constant)
        P_fixed = 5
        train_times_N = []
        predict_times_N = []
        
        for N in N_values:
            train_time_sum = 0
            predict_time_sum = 0
            
            for _ in range(num_average_time):
                X, y = create_fake_data(N, P_fixed, case_type)
                
                # Measure training time
                tree = DecisionTree(criterion="information_gain", max_depth=5)
                start_time = time.time()
                tree.fit(X, y)
                train_time_sum += time.time() - start_time
                
                # Measure prediction time
                start_time = time.time()
                tree.predict(X)
                predict_time_sum += time.time() - start_time
            
            train_times_N.append(train_time_sum / num_average_time)
            predict_times_N.append(predict_time_sum / num_average_time)
        
        # Vary P (keeping N constant)
        N_fixed = 100
        train_times_P = []
        predict_times_P = []
        
        for P in P_values:
            train_time_sum = 0
            predict_time_sum = 0
            
            for _ in range(num_average_time):
                X, y = create_fake_data(N_fixed, P, case_type)
                
                # Measure training time
                tree = DecisionTree(criterion="information_gain", max_depth=5)
                start_time = time.time()
                tree.fit(X, y)
                train_time_sum += time.time() - start_time
                
                # Measure prediction time
                start_time = time.time()
                tree.predict(X)
                predict_time_sum += time.time() - start_time
            
            train_times_P.append(train_time_sum / num_average_time)
            predict_times_P.append(predict_time_sum / num_average_time)
        
        return train_times_N, predict_times_N, train_times_P, predict_times_P

    def plot_results(N_values, P_values, results_dict):
        """Plot the timing results"""
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Decision Tree Runtime Complexity Analysis', fontsize=16)
        
        cases = ["discrete_discrete", "discrete_real", "real_discrete", "real_real"]
        case_names = ["Discrete→Discrete", "Discrete→Real", "Real→Discrete", "Real→Real"]
        
        for i, (case, name) in enumerate(zip(cases, case_names)):
            train_times_N, predict_times_N, train_times_P, predict_times_P = results_dict[case]
            
            # Training time vs N
            axes[0, i].plot(N_values, train_times_N, 'o-', color='blue')
            axes[0, i].set_xlabel('Number of Samples (N)')
            axes[0, i].set_ylabel('Training Time (seconds)')
            axes[0, i].set_title(f'{name}\nTraining Time vs N')
            axes[0, i].grid(True)
            
            # Prediction time vs P
            axes[1, i].plot(P_values, train_times_P, 's-', color='red')
            axes[1, i].set_xlabel('Number of Features (P)')
            axes[1, i].set_ylabel('Training Time (seconds)')
            axes[1, i].set_title(f'{name}\nTraining Time vs P')
            axes[1, i].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Create another plot for prediction times
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle('Decision Tree Prediction Time Analysis', fontsize=16)
        
        for i, (case, name) in enumerate(zip(cases, case_names)):
            train_times_N, predict_times_N, train_times_P, predict_times_P = results_dict[case]
            
            axes[i].plot(N_values, predict_times_N, 'o-', color='green', label='vs N')
            axes[i].set_xlabel('Number of Samples (N)')
            axes[i].set_ylabel('Prediction Time (seconds)')
            axes[i].set_title(f'{name}\nPrediction Time vs N')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()

    # Run experiments
    print("=== Decision Tree Runtime Complexity Analysis ===")
    print("Running experiments... This may take a few minutes.")

    N_values = [50, 100, 200, 400, 800]
    P_values = [2, 4, 8, 16, 32]
    cases = ["discrete_discrete", "discrete_real", "real_discrete", "real_real"]

    results = {}

    for case in cases:
        print(f"\nRunning experiments for {case}...")
        results[case] = measure_time(N_values, P_values, case)

    # Plot results
    plot_results(N_values, P_values, results)

    # Print theoretical analysis
    print("\n=== Theoretical Time Complexity Analysis ===")
    print("Decision Tree Training:")
    print("- Time Complexity: O(N * P * log(N)) where N=samples, P=features")
    print("- At each node, we evaluate P features")
    print("- For each feature, we sort N samples: O(N log N)")
    print("- Tree depth is typically O(log N)")
    print("\nDecision Tree Prediction:")
    print("- Time Complexity: O(log N) per sample")
    print("- Just traverse from root to leaf")
    print("- For N samples: O(N * log N)")

    # Analyze results
    print("\n=== Empirical Results Analysis ===")
    for case in cases:
        train_times_N, predict_times_N, train_times_P, predict_times_P = results[case]
        print(f"\n{case.upper()}:")
        print(f"Training time growth with N: {train_times_N[-1]/train_times_N[0]:.2f}x")
        print(f"Training time growth with P: {train_times_P[-1]/train_times_P[0]:.2f}x")
        print(f"Prediction time growth with N: {predict_times_N[-1]/predict_times_N[0]:.2f}x")