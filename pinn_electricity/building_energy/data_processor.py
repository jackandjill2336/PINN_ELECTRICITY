# pinn_electricity/building_energy/data_processor.py
"""
SERL Energy Data Loading and Preprocessing
Based on proven workflow for building energy consumption prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SERLDataProcessor:
    """
    Data processor for SERL energy consumption datasets
    Implements exact preprocessing pipeline from proven workflow
    """
    
    def __init__(self, min_energy_threshold: float = 0.005):
        """
        Initialize SERL data processor
        
        Args:
            min_energy_threshold: Minimum energy value to keep (kWh)
        """
        self.min_energy_threshold = min_energy_threshold
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.original_shape = None
        self.cleaned_shape = None
        
    def load_serl_data(self, filepath: str) -> pd.DataFrame:
        """
        Load SERL CSV data with initial inspection
        
        Args:
            filepath: Path to SERL CSV file
            
        Returns:
            Raw dataframe
        """
        print("Loading SERL energy data...")
        df = pd.read_csv(filepath)
        
        self.original_shape = df.shape
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert energy units from Wh to kWh
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with converted units
        """
        df_converted = df.copy()
        
        # Check if unit conversion is needed
        if 'unit' in df_converted.columns:
            units = df_converted['unit'].unique()
            print(f"Found units: {units}")
            
            if any('Wh' in str(unit) for unit in units):
                df_converted['value_kWh'] = df_converted['value'] / 1000
                print("Converted Wh to kWh")
            else:
                df_converted['value_kWh'] = df_converted['value']
        else:
            df_converted['value_kWh'] = df_converted['value']
            
        return df_converted
    
    def parse_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and create temporal features from aggregation_period
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with temporal features
        """
        df_temporal = df.copy()
        
        # Parse aggregation period to datetime
        df_temporal['date'] = pd.to_datetime(df_temporal['aggregation_period'], errors='coerce')
        
        # Extract temporal components
        df_temporal['month'] = df_temporal['date'].dt.month
        df_temporal['year'] = df_temporal['date'].dt.year
        df_temporal['day_of_year'] = df_temporal['date'].dt.dayofyear
        
        # Create cyclical features for seasonal patterns
        df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
        df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)
        df_temporal['day_sin'] = np.sin(2 * np.pi * df_temporal['day_of_year'] / 365)
        df_temporal['day_cos'] = np.cos(2 * np.pi * df_temporal['day_of_year'] / 365)
        
        # Create season categorical variable
        def get_season(month):
            if month in [12, 1, 2]: 
                return 'Winter'
            elif month in [3, 4, 5]: 
                return 'Spring'
            elif month in [6, 7, 8]: 
                return 'Summer'
            else: 
                return 'Autumn'
        
        df_temporal['season'] = df_temporal['month'].apply(get_season)
        
        print("Created temporal features: month, year, cyclical encodings, seasons")
        
        return df_temporal
    
    def encode_building_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode building segment categorical variables
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with encoded building segments
        """
        df_encoded = df.copy()
        
        # Check for different segment column patterns
        if 'segment_1_value' in df_encoded.columns:
            df_encoded['building_segment'] = pd.factorize(df_encoded['segment_1_value'])[0]
            print(f"Encoded segment_1_value: {df_encoded['building_segment'].nunique()} unique segments")
        elif 'segment_2_value' in df_encoded.columns:
            df_encoded['building_segment'] = pd.factorize(df_encoded['segment_2_value'])[0]
            print(f"Encoded segment_2_value: {df_encoded['building_segment'].nunique()} unique segments")
        else:
            df_encoded['building_segment'] = 0  # Default single segment
            print("No segment columns found, using default segment")
            
        return df_encoded
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing missing values and outliers
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        print("Cleaning data...")
        
        # Remove rows with missing critical values
        required_columns = ['mean_temp', 'mean_hdd', 'value_kWh']
        df_clean = df_clean.dropna(subset=required_columns)
        
        print(f"Removed {initial_count - len(df_clean)} rows with missing values")
        
        # Remove suspiciously small energy values (likely measurement errors)
        df_clean = df_clean[df_clean['value_kWh'] > self.min_energy_threshold]
        
        print(f"Removed values <= {self.min_energy_threshold} kWh")
        
        # Remove extreme outliers using IQR method
        Q1 = df_clean['value_kWh'].quantile(0.25)
        Q3 = df_clean['value_kWh'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = len(df_clean)
        df_clean = df_clean[(df_clean['value_kWh'] >= lower_bound) & 
                           (df_clean['value_kWh'] <= upper_bound)]
        
        print(f"Removed {outliers_before - len(df_clean)} outliers using IQR method")
        
        self.cleaned_shape = df_clean.shape
        
        print(f"Final cleaned data shape: {df_clean.shape}")
        print(f"Data retention: {len(df_clean)/initial_count*100:.1f}%")
        
        return df_clean
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate comprehensive data summary
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'total_samples': len(df),
            'target_stats': df['value_kWh'].describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None
            },
            'categorical_counts': {
                'seasons': df['season'].value_counts().to_dict() if 'season' in df.columns else None,
                'building_segments': df['building_segment'].value_counts().to_dict() if 'building_segment' in df.columns else None
            }
        }
        
        return summary
    
    def process_serl_dataset(self, filepath: str, verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
        """
        Complete SERL data processing pipeline
        
        Args:
            filepath: Path to SERL CSV file
            verbose: Whether to print processing steps
            
        Returns:
            Tuple of (processed_dataframe, summary_stats)
        """
        if verbose:
            print("Starting SERL data processing pipeline...")
            
        # Step 1: Load data
        df = self.load_serl_data(filepath)
        
        # Step 2: Convert units
        df = self.convert_units(df)
        
        # Step 3: Parse temporal features
        df = self.parse_temporal_features(df)
        
        # Step 4: Encode building segments
        df = self.encode_building_segments(df)
        
        # Step 5: Clean data
        df_clean = self.clean_data(df)
        
        # Step 6: Generate summary
        summary = self.get_data_summary(df_clean)
        
        if verbose:
            print("\nData processing complete!")
            print(f"Original shape: {self.original_shape}")
            print(f"Final shape: {self.cleaned_shape}")
            print(f"Target variable range: {summary['target_stats']['min']:.3f} to {summary['target_stats']['max']:.3f} kWh")
            
        return df_clean, summary
    
    def prepare_features_target(self, df: pd.DataFrame, 
                               feature_columns: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable
        
        Args:
            df: Processed dataframe
            feature_columns: List of feature columns to use
            
        Returns:
            Tuple of (features_dataframe, target_series)
        """
        if feature_columns is None:
            # Default feature set based on proven approach
            feature_columns = ['mean_temp', 'mean_hdd', 'month', 'building_segment']
            
            # Add cyclical features if available
            cyclical_features = ['month_sin', 'month_cos', 'day_sin', 'day_cos']
            for feat in cyclical_features:
                if feat in df.columns:
                    feature_columns.append(feat)
        
        # Select features that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) == 0:
            raise ValueError("No specified features found in dataframe")
            
        print(f"Using features: {available_features}")
        
        X = df[available_features].copy()
        y = df['value_kWh'].copy()
        
        # Remove any remaining missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        return X, y
    
    def scale_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   y_train: pd.Series, y_test: pd.Series, 
                   scale_features: bool = True, 
                   scale_target: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features and target variables
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Target variables
            scale_features: Whether to scale features
            scale_target: Whether to scale target
            
        Returns:
            Tuple of scaled arrays (X_train, X_test, y_train, y_test)
        """
        print(f"Scaling data - Features: {scale_features}, Target: {scale_target}")
        
        # Scale features
        if scale_features:
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
            
        # Scale target
        if scale_target:
            y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        else:
            y_train_scaled = y_train.values
            y_test_scaled = y_test.values
            
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    
    def inverse_transform_predictions(self, y_pred_scaled: np.ndarray, 
                                    scale_target: bool = True) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale
        
        Args:
            y_pred_scaled: Scaled predictions
            scale_target: Whether target was scaled during training
            
        Returns:
            Predictions in original scale
        """
        if scale_target and hasattr(self.scaler_y, 'inverse_transform'):
            return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        else:
            return y_pred_scaled.flatten()

# Example usage function
def load_and_process_serl_data(filepath: str, 
                              min_energy_threshold: float = 0.005,
                              feature_columns: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Convenience function to load and process SERL data
    
    Args:
        filepath: Path to SERL CSV file
        min_energy_threshold: Minimum energy threshold for cleaning
        feature_columns: Specific features to extract
        
    Returns:
        Tuple of (features, target, summary_stats)
    """
    processor = SERLDataProcessor(min_energy_threshold=min_energy_threshold)
    
    # Process dataset
    df_clean, summary = processor.process_serl_dataset(filepath)
    
    # Prepare features and target
    X, y = processor.prepare_features_target(df_clean, feature_columns)
    
    return X, y, summary

if __name__ == "__main__":
    # Example usage
    print("SERL Data Processor")
    print("Example usage:")
    print("processor = SERLDataProcessor()")
    print("df_clean, summary = processor.process_serl_dataset('your_file.csv')")
    print("X, y = processor.prepare_features_target(df_clean)")
