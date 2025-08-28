import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class AndroidDataProcessor:
    """
    Process Android phone sensor data for Human Activity Recognition
    """
    
    def __init__(self, data_dir: str = "android_data", target_sample_rate: float = 50.0, original_sample_rate: float = 400.0):
        """
        Initialize the processor
        
        Args:
            data_dir: Directory containing the CSV files
            target_sample_rate: Target sampling rate (Hz) to match UCI HAR dataset
            original_sample_rate: Original sample rate of your Android data
        """
        self.data_dir = Path(data_dir)
        self.target_sample_rate = target_sample_rate
        self.original_sample_rate = original_sample_rate
        self.activity_mapping = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}
        
    def load_and_clean_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and clean a single CSV file
        """
        try:
            # Read the CSV file, skipping the header comments
            df = pd.read_csv(filepath)
            
            # Remove any rows with missing values
            df = df.dropna()
            
            # Convert time to seconds if needed
            if df['time'].max() > 1000:  # Assume milliseconds
                df['time'] = df['time'] / 1000
            
            # Extract the sensor columns (assuming gyroscope data)
            # Rename columns to match accelerometer format for consistency
            sensor_columns = ['ωₓ (rad/s)', 'ωᵧ (rad/s)', 'ωᶻ (rad/s)']
            if all(col in df.columns for col in sensor_columns):
                df = df[['time'] + sensor_columns]
                df.columns = ['time', 'accx', 'accy', 'accz']  # Rename for consistency
            else:
                # Try alternative column names
                possible_columns = [col for col in df.columns if any(x in col.lower() for x in ['x', 'y', 'z', 'acc', 'gyr'])]
                if len(possible_columns) >= 3:
                    df = df[['time'] + possible_columns[:3]]
                    df.columns = ['time', 'accx', 'accy', 'accz']
                else:
                    raise ValueError(f"Cannot identify sensor columns in {filepath}")
            
            return df
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data from 400Hz to target sampling rate (50Hz)
        """
        # Set time as index
        df = df.set_index('time')
        
        # Calculate actual sampling rate from data
        time_diff = df.index.diff().dropna()
        actual_rate = 1 / time_diff.median()
        print(f"Detected sampling rate: {actual_rate:.2f} Hz")
        
        # For 400Hz data, we can use every 8th sample to get 50Hz (400/8 = 50)
        downsample_factor = int(self.original_sample_rate / self.target_sample_rate)
        
        if abs(actual_rate - self.original_sample_rate) < 50:  # Within 50Hz tolerance
            print(f"Using downsampling: taking every {downsample_factor} samples")
            # Simple downsampling - more efficient and preserves signal characteristics
            df_resampled = df.iloc[::downsample_factor].copy()
        else:
            print(f"Using interpolation resampling")
            # Fallback to interpolation method
            start_time = df.index.min()
            end_time = df.index.max()
            new_time_index = np.arange(start_time, end_time, 1/self.target_sample_rate)
            
            df_resampled = df.reindex(df.index.union(new_time_index)).interpolate(method='linear')
            df_resampled = df_resampled.reindex(new_time_index)
        
        return df_resampled.reset_index()
    
    def pad_data(self, data: np.ndarray, target_length: int, method: str = 'reflect') -> np.ndarray:
        """
        Pad data to target length using various methods
        
        Args:
            data: Input data array (n_samples, n_features)
            target_length: Desired length
            method: Padding method ('reflect', 'repeat', 'symmetric', 'mean')
        
        Returns:
            Padded data array
        """
        current_length = len(data)
        
        if current_length >= target_length:
            return data[:target_length]
        
        padding_needed = target_length - current_length
        
        if method == 'reflect':
            # Reflect the signal at boundaries (good for periodic signals like WALKING)
            if current_length > 1:
                padded_data = np.pad(data, ((0, padding_needed), (0, 0)), mode='reflect')
            else:
                padded_data = np.tile(data, (target_length, 1))
                
        elif method == 'repeat':
            # Repeat the entire signal
            repetitions = int(np.ceil(target_length / current_length))
            repeated = np.tile(data, (repetitions, 1))
            padded_data = repeated[:target_length]
            
        elif method == 'symmetric':
            # Symmetric padding (mirror without repeating edge values)
            if current_length > 2:
                padded_data = np.pad(data, ((0, padding_needed), (0, 0)), mode='symmetric')
            else:
                padded_data = np.tile(data, (target_length, 1))
                
        elif method == 'mean':
            # Pad with mean values (good for static activities)
            mean_values = np.mean(data, axis=0)
            padding = np.tile(mean_values, (padding_needed, 1))
            padded_data = np.vstack([data, padding])
            
        else:
            # Default: repeat last value
            last_value = data[-1]
            padding = np.tile(last_value, (padding_needed, 1))
            padded_data = np.vstack([data, padding])
        
        return padded_data
    
    def determine_padding_method(self, data: np.ndarray, activity_name: str) -> str:
        """
        Determine the best padding method based on activity type and data characteristics
        """
        activity_lower = activity_name.lower()
        
        # Calculate signal variance to determine if it's dynamic or static
        variance = np.var(data, axis=0).mean()
        
        if 'walk' in activity_lower:
            return 'reflect'  # WALKING is periodic, reflection works well
        elif activity_lower in ['sitting', 'standing', 'laying']:
            if variance < 0.1:  # Low variance indicates static activity
                return 'mean'
            else:
                return 'symmetric'
        else:
            return 'reflect'  # Default for unknown activities
    def extract_trials(self, df: pd.DataFrame, trial_specs: List[Tuple[float, float]], activity_name: str = 'unknown') -> List[np.ndarray]:
        """
        Extract individual trials from continuous data with intelligent padding
        
        Args:
            df: DataFrame with sensor data
            trial_specs: List of (start_time, end_time) tuples for each trial
            activity_name: Name of the activity for intelligent padding
        
        Returns:
            List of trial data arrays, each shaped (target_samples, 3)
        """
        trials = []
        target_samples = int(10 * self.target_sample_rate)  # 10 seconds at target rate
        min_samples = int(3 * self.target_sample_rate)      # Minimum 3 seconds of data
        
        for i, (start_time, end_time) in enumerate(trial_specs):
            # Extract trial data
            mask = (df['time'] >= start_time) & (df['time'] <= end_time)
            trial_data = df[mask][['accx', 'accy', 'accz']].values
            
            if len(trial_data) < min_samples:
                print(f"Trial {i+1} from {start_time:.2f}s to {end_time:.2f}s too short: {len(trial_data)} samples (min: {min_samples})")
                continue
            
            if len(trial_data) < target_samples:
                # Need to pad the data
                padding_method = self.determine_padding_method(trial_data, activity_name)
                print(f"Trial {i+1}: Padding {len(trial_data)} samples to {target_samples} using '{padding_method}' method")
                trial_data = self.pad_data(trial_data, target_samples, padding_method)
            elif len(trial_data) > target_samples:
                # Truncate if too long
                print(f"Trial {i+1}: Truncating {len(trial_data)} samples to {target_samples}")
                trial_data = trial_data[:target_samples]
            
            trials.append(trial_data)
            print(f"Final trial {i+1} shape: {trial_data.shape}")
        
        return trials
    
    def process_activity_file(self, activity_name: str, trial_specs: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single activity file and extract trials
        
        Args:
            activity_name: Name of the activity (e.g., 'WALKING')
            trial_specs: List of (start_time, end_time) for trials
        
        Returns:
            X: Array of shape (n_trials, n_samples, 3)
            y: Array of activity labels
        """
        # Find the file
        filepath = self.data_dir +f'{activity_name}.csv'
        
        if filepath is None:
            print(f"Could not find file for activity: {activity_name}")
            return np.array([]), np.array([])
        
        print(f"\nProcessing {activity_name} from {filepath}")
        
        # Load and process data
        df = self.load_and_clean_data(filepath)
        if df is None:
            return np.array([]), np.array([])
        
        # Resample to target rate
        df = self.resample_data(df)
        
        # Extract trials
        trials = self.extract_trials(df, trial_specs, activity_name)
        
        if not trials:
            print(f"No valid trials extracted for {activity_name}")
            return np.array([]), np.array([])
        
        # Convert to arrays
        X = np.array(trials)
        y = np.full(len(trials), self.activity_mapping[activity_name.lower()])
        
        print(f"Final data shape for {activity_name}: {X.shape}")
        return X, y
    
    def visualize_data(self, df: pd.DataFrame, title: str = "Sensor Data"):
        """
        Visualize the sensor data
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        axes[0].plot(df['time'], df['accx'])
        axes[0].set_ylabel('X-axis')
        axes[0].set_title(f'{title} - Raw Data')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df['time'], df['accy'])
        axes[1].set_ylabel('Y-axis')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(df['time'], df['accz'])
        axes[2].set_ylabel('Z-axis')
        axes[2].set_xlabel('Time (s)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_dataset(self, activity_trials: Dict[str, List[Tuple[float, float]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create complete dataset from all activities
        
        Args:
            activity_trials: Dictionary mapping activity names to list of (start, end) times
            
        Returns:
            X: Complete feature array (n_samples, n_timepoints, 3)
            y: Complete label array (n_samples,)
        """
        all_X = []
        all_y = []
        
        for activity_name, trial_specs in activity_trials.items():
            X, y = self.process_activity_file(activity_name, trial_specs)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        
        if not all_X:
            print("No data was successfully processed!")
            return np.array([]), np.array([])
        
        # Combine all data
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        print(f"\nFinal dataset shape: {X_combined.shape}")
        print(f"Label distribution: {np.bincount(y_combined)}")
        
        return X_combined, y_combined


def main():
    """
    Example usage of the AndroidDataProcessor for 400Hz data
    """
    
    # Initialize processor for 400Hz input data
    processor = AndroidDataProcessor(
        data_dir="", 
        target_sample_rate=50.0,
        original_sample_rate=400.0
    )
    
    # Define your trial specifications for each activity
    # Format: activity_name -> [(start_time1, end_time1), (start_time2, end_time2), ...]
    # Note: With 400Hz data, you can have shorter intervals and still get good results
    activity_trials = {
        'WALKING': [
            (0, 10.0),
            (10.0, 20.0),
            (20.0, 30.0),
            (30.0, 40.0),
            (40.0, 50.0)
        ],
        'SITTING': [
            (0, 15.0),
            (30.0, 45.0),
            (60.0, 75.0),
            (90.0, 105.0)
        ],
        'STANDING': [
            (15., 30.0),
            (45.0, 60.0),
            (75.0, 90.0)
        ],
        'LAYING': [
            (0, 15.0),
            (30.0, 45.0),
            (60.0, 75.0),
            (90.0, 105.0)
        ],
        'WALKING_DOWNSTAIRS': [
            (0, 10.0),
            (10.0, 20.0),
            (20.0, 30.0),
            (30.0, 40.0),
            (40.0, 50.0)
        ],
        'WALKING_UPSTAIRS': [
            (0, 10.0),
            (10.0, 20.0),
            (20.0, 30.0),
            (30.0, 40.0),
            (40.0, 50.0)
        ],
        # Add more activities as needed
    }
    
    # Process all data
    X, y = processor.create_dataset(activity_trials)
    
    if len(X) == 0:
        print("No data processed. Please check your files and trial specifications.")
        return
    
    # Save processed data
    np.save('android_X.npy', X)
    np.save('android_y.npy', y)
    print(f"\nProcessed data saved:")
    print(f"X shape: {X.shape} -> 'android_X.npy'")
    print(f"y shape: {y.shape} -> 'android_y.npy'")
    
    # Example: Test with your decision tree
    # Assuming you have your decision tree implementation available
    try:
        from tree.base import DecisionTree
        from metrics import accuracy
        from sklearn.model_selection import train_test_split
        
        # Flatten data for decision tree input
        X_flat = X.reshape(X.shape[0], -1)
        X_df = pd.DataFrame(X_flat)
        y_series = pd.Series(y, dtype='category')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.3, random_state=42, stratify=y_series
        )
        
        # Train decision tree
        dt = DecisionTree(criterion="information_gain", max_depth=6)
        dt.fit(X_train, y_train)
        
        # Make predictions
        y_pred = dt.predict(X_test)
        
        # Calculate accuracy
        acc = accuracy(y_pred, y_test)
        print(f"\nDecision Tree Accuracy on Android data: {acc:.4f}")
        
        # Print per-class performance
        activity_names = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 
                         4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}
        
        for class_id in np.unique(y_test):
            if class_id in activity_names:
                from metrics import precision, recall
                prec = precision(y_pred, y_test, class_id)
                rec = recall(y_pred, y_test, class_id)
                print(f"{activity_names[class_id]}: Precision={prec:.4f}, Recall={rec:.4f}")
        
    except ImportError:
        print("Decision tree modules not found. Data has been processed and saved.")
    
    return X, y


if __name__ == "__main__":
    # Run the main processing function
    X, y = main()
    
    # Additional visualization if data exists
    if len(X) > 0:
        # Plot sample from each activity
        activity_names = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 
                         4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (class_id, name) in enumerate(activity_names.items()):
            if i < 6 and class_id in y:
                # Find first sample of this class
                sample_idx = np.where(y == class_id)[0]
                if len(sample_idx) > 0:
                    sample = X[sample_idx[0]]
                    
                    # Plot all three axes
                    time_axis = np.arange(len(sample)) / 50.0  # 50Hz sampling rate
                    axes[i].plot(time_axis, sample[:, 0], label='X-axis', alpha=0.7)
                    axes[i].plot(time_axis, sample[:, 1], label='Y-axis', alpha=0.7)
                    axes[i].plot(time_axis, sample[:, 2], label='Z-axis', alpha=0.7)
                    
                    axes[i].set_title(f'{name}')
                    axes[i].set_xlabel('Time (s)')
                    axes[i].set_ylabel('Sensor Reading')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()