# pinn_electricity/building_energy/comprehensive_analysis.py
"""
Comprehensive Analysis Framework for Large-Scale Energy Datasets
Combines all proven approaches: PINN, baseline models, and interpretability analysis
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LargeDatasetProcessor:
    """
    Large dataset processor for SERL energy consumption data
    Handles sampling, cleaning, and feature engineering for large datasets
    """
    
    def __init__(self, sample_size: int = 50000, random_state: int = 42):
        self.sample_size = sample_size
        self.random_state = random_state
        self.feature_names = None
        self.encoders = {}
        
    def create_sample_dataset(self, filepath: str, output_path: str = "energy_sample_large.csv") -> pd.DataFrame:
        """
        Create manageable sample from large dataset
        """
        print("Loading large dataset...")
        df_large = pd.read_csv(filepath)
        print(f"Original size: {df_large.shape}")
        
        # Create stratified sample
        actual_sample_size = min(self.sample_size, len(df_large) // 5)
        df_sample = df_large.sample(n=actual_sample_size, random_state=self.random_state)
        print(f"Sample size: {df_sample.shape}")
        
        # Save sample
        df_sample.to_csv(output_path, index=False)
        print(f"Sample saved to: {output_path}")
        print(f"Sample file size: {df_sample.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df_sample
    
    def analyze_dataset_structure(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive dataset structure analysis
        """
        print("=== DATASET STRUCTURE ANALYSIS ===")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Column analysis
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            print(f"{i:2d}. {col:30} | {str(dtype):10} | Missing: {missing:5d} | Unique: {unique:5d}")
        
        # Identify potential targets and features
        target_candidates = [col for col in df.columns 
                           if any(keyword in col.lower() 
                                for keyword in ['energy', 'kwh', 'value', 'consumption', 'mean'])]
        
        feature_candidates = [col for col in df.columns 
                            if any(keyword in col.lower() 
                                 for keyword in ['temp', 'hdd', 'solar', 'segment', 'quantity', 'unit'])]
        
        analysis = {
            'shape': df.shape,
            'target_candidates': target_candidates,
            'feature_candidates': feature_candidates,
            'missing_summary': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict()
        }
        
        print(f"\nTarget candidates: {target_candidates}")
        print(f"Feature candidates: {feature_candidates}")
        
        return analysis
    
    def clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning and feature engineering based on proven approach
        """
        print("=== DATA CLEANING AND FEATURE ENGINEERING ===")
        df_clean = df.copy()
        
        # Step 1: Handle missing values strategically
        print(f"Original shape: {df_clean.shape}")
        
        # Remove rows where target variable is missing
        df_clean = df_clean.dropna(subset=['mean'])
        print(f"After removing missing target: {df_clean.shape}")
        
        # Step 2: Filter for relevant energy consumption data
        if 'quantity' in df_clean.columns:
            consumption_quantities = ['Electricity net', 'Gas', 'Electricity']
            df_clean = df_clean[df_clean['quantity'].isin(consumption_quantities)]
            print(f"After filtering quantities: {df_clean.shape}")
        
        if 'unit' in df_clean.columns:
            meaningful_units = ['kWh/day', 'kWh/m2/day', 'kWh/person/day']
            df_clean = df_clean[df_clean['unit'].isin(meaningful_units)]
            print(f"After filtering units: {df_clean.shape}")
        
        # Step 3: Remove rows with missing weather data
        weather_columns = ['mean_temp', 'mean_hdd', 'mean_solar']
        available_weather = [col for col in weather_columns if col in df_clean.columns]
        df_clean = df_clean.dropna(subset=available_weather)
        print(f"After removing missing weather data: {df_clean.shape}")
        
        # Step 4: Feature Engineering
        print("Creating engineered features...")
        
        # Parse time information
        if 'aggregation_period' in df_clean.columns:
            df_clean['year'] = df_clean['aggregation_period'].str[:4].astype(float)
            df_clean['month'] = pd.to_datetime(df_clean['aggregation_period'], errors='coerce').dt.month
            df_clean['month'] = df_clean['month'].fillna(6)  # Default to mid-year
        
        # Encode categorical variables
        categorical_columns = ['quantity', 'unit', 'segment_1_value', 'segment_2_value']
        for col in categorical_columns:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].fillna('Unknown'))
                self.encoders[col] = le
        
        # Physics-informed features
        if 'mean_temp' in df_clean.columns and 'mean_hdd' in df_clean.columns:
            df_clean['temp_hdd_interaction'] = df_clean['mean_temp'] * df_clean['mean_hdd']
            df_clean['temp_squared'] = df_clean['mean_temp'] ** 2
            df_clean['hdd_squared'] = df_clean['mean_hdd'] ** 2
            
        if 'mean_solar' in df_clean.columns and 'mean_temp' in df_clean.columns:
            df_clean['solar_temp_interaction'] = df_clean['mean_solar'] * df_clean['mean_temp']
        
        # Seasonal features
        if 'month' in df_clean.columns:
            df_clean['season_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
            df_clean['season_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)
        
        # Temperature zones (physics-based)
        if 'mean_temp' in df_clean.columns:
            df_clean['very_cold'] = (df_clean['mean_temp'] < 5).astype(float)
            df_clean['cold'] = ((df_clean['mean_temp'] >= 5) & (df_clean['mean_temp'] < 15)).astype(float)
            df_clean['mild'] = ((df_clean['mean_temp'] >= 15) & (df_clean['mean_temp'] < 20)).astype(float)
            df_clean['warm'] = (df_clean['mean_temp'] >= 20).astype(float)
        
        # High energy demand indicators
        if 'mean_hdd' in df_clean.columns:
            df_clean['high_hdd'] = (df_clean['mean_hdd'] > df_clean['mean_hdd'].quantile(0.75)).astype(float)
            
        if 'mean_solar' in df_clean.columns:
            df_clean['low_solar'] = (df_clean['mean_solar'] < df_clean['mean_solar'].quantile(0.25)).astype(float)
        
        print(f"Created {df_clean.shape[1]} total features")
        
        return df_clean
    
    def prepare_modeling_data(self, df_clean: pd.DataFrame, target_col: str = 'mean') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare final modeling dataset
        """
        # Define feature sets
        basic_features = []
        
        # Add available basic features
        basic_candidates = ['mean_temp', 'mean_hdd', 'mean_solar', 'month']
        for feat in basic_candidates:
            if feat in df_clean.columns:
                basic_features.append(feat)
        
        # Add encoded categorical features
        encoded_features = [col for col in df_clean.columns if col.endswith('_encoded')]
        basic_features.extend(encoded_features)
        
        # Physics features
        physics_features = []
        physics_candidates = [
            'temp_hdd_interaction', 'temp_squared', 'hdd_squared', 'solar_temp_interaction',
            'season_sin', 'season_cos', 'very_cold', 'cold', 'mild', 'warm',
            'high_hdd', 'low_solar'
        ]
        for feat in physics_candidates:
            if feat in df_clean.columns:
                physics_features.append(feat)
        
        all_features = basic_features + physics_features
        
        print(f"Basic features ({len(basic_features)}): {basic_features}")
        print(f"Physics features ({len(physics_features)}): {physics_features}")
        print(f"Total features: {len(all_features)}")
        
        # Prepare final dataset
        X = df_clean[all_features]
        y = df_clean[target_col]
        
        # Remove any remaining missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask].values
        y_clean = y[mask].values
        
        print(f"Final dataset shape: {X_clean.shape}")
        print(f"Target range: {y_clean.min():.4f} to {y_clean.max():.4f}")
        
        self.feature_names = all_features
        
        return X_clean, y_clean, all_features

class ComprehensiveModelComparison:
    """
    Comprehensive model comparison framework
    Tests Random Forest, Standard NN, and PINN approaches
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}
        self.models = {}
        self.scaler = StandardScaler()
        
    def train_random_forest_baseline(self, X_train: np.ndarray, X_test: np.ndarray, 
                                   y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train Random Forest baseline
        """
        print("=== RANDOM FOREST BASELINE ===")
        
        rf = RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)
        
        print(f"Random Forest: R² = {rf_r2:.4f}, RMSE = {rf_rmse:.3f}, MAE = {rf_mae:.3f}")
        
        self.models['random_forest'] = rf
        
        return {
            'r2': rf_r2,
            'rmse': rf_rmse,
            'mae': rf_mae,
            'predictions': rf_pred,
            'feature_importance': rf.feature_importances_
        }
    
    def train_neural_network_variants(self, X_train: np.ndarray, X_test: np.ndarray,
                                    y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train multiple neural network variants
        """
        print("=== NEURAL NETWORK VARIANTS ===")
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        nn_results = {}
        
        # Basic NN
        print("Training Basic NN...")
        basic_nn = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ], name='BasicNN')
        
        basic_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        basic_nn.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
        
        basic_pred = basic_nn.predict(X_test_scaled, verbose=0).flatten()
        basic_r2 = r2_score(y_test, basic_pred)
        
        nn_results['basic'] = {
            'r2': basic_r2,
            'rmse': np.sqrt(mean_squared_error(y_test, basic_pred)),
            'mae': mean_absolute_error(y_test, basic_pred),
            'predictions': basic_pred
        }
        
        print(f"Basic NN: R² = {basic_r2:.4f}")
        
        # Deeper NN
        print("Training Deep NN...")
        deep_nn = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.1),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ], name='DeepNN')
        
        deep_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        deep_nn.fit(X_train_scaled, y_train, epochs=150, batch_size=32, verbose=0)
        
        deep_pred = deep_nn.predict(X_test_scaled, verbose=0).flatten()
        deep_r2 = r2_score(y_test, deep_pred)
        
        nn_results['deep'] = {
            'r2': deep_r2,
            'rmse': np.sqrt(mean_squared_error(y_test, deep_pred)),
            'mae': mean_absolute_error(y_test, deep_pred),
            'predictions': deep_pred
        }
        
        print(f"Deep NN: R² = {deep_r2:.4f}")
        
        # Ensemble NN
        print("Training NN Ensemble...")
        ensemble_predictions = []
        
        for i in range(3):
            ensemble_nn = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            ensemble_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            ensemble_nn.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
            
            pred = ensemble_nn.predict(X_test_scaled, verbose=0).flatten()
            ensemble_predictions.append(pred)
        
        ensemble_pred = np.mean(ensemble_predictions, axis=0)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        nn_results['ensemble'] = {
            'r2': ensemble_r2,
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'predictions': ensemble_pred
        }
        
        print(f"NN Ensemble: R² = {ensemble_r2:.4f}")
        
        # Store models
        self.models['basic_nn'] = basic_nn
        self.models['deep_nn'] = deep_nn
        
        return nn_results
    
    def train_physics_informed_nn(self, X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray,
                                feature_names: List[str]) -> Dict:
        """
        Train Physics-Informed Neural Network
        """
        print("=== PHYSICS-INFORMED NEURAL NETWORK ===")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build PINN architecture
        inputs = Input(shape=(X_train.shape[1],), name='inputs')
        
        # Physics-aware branches
        temp_branch = Dense(16, activation='tanh', name='temp_branch')(inputs)
        temp_branch = Dense(8, activation='tanh')(temp_branch)
        
        hdd_branch = Dense(16, activation='relu', name='hdd_branch')(inputs)
        hdd_branch = Dense(8, activation='relu')(hdd_branch)
        
        solar_branch = Dense(12, activation='sigmoid', name='solar_branch')(inputs)
        solar_branch = Dense(6, activation='sigmoid')(solar_branch)
        
        seasonal_branch = Dense(8, activation='tanh', name='seasonal_branch')(inputs)
        
        # Combine physics branches
        physics_combined = Concatenate(name='physics_combine')([
            temp_branch, hdd_branch, solar_branch, seasonal_branch
        ])
        
        # Main processing
        x = Dense(64, activation='relu', name='main_dense1')(physics_combined)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu', name='main_dense2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        x = Dense(16, activation='relu', name='main_dense3')(x)
        output = Dense(1, name='energy_output')(x)
        
        pinn_model = Model(inputs=inputs, outputs=output, name='PINN')
        
        # Compile and train
        pinn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        pinn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, verbose=0)
        
        # Evaluate
        pinn_pred = pinn_model.predict(X_test_scaled, verbose=0).flatten()
        pinn_r2 = r2_score(y_test, pinn_pred)
        
        print(f"PINN: R² = {pinn_r2:.4f}")
        
        self.models['pinn'] = pinn_model
        
        return {
            'r2': pinn_r2,
            'rmse': np.sqrt(mean_squared_error(y_test, pinn_pred)),
            'mae': mean_absolute_error(y_test, pinn_pred),
            'predictions': pinn_pred
        }
    
    def run_comprehensive_comparison(self, X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str], test_size: float = 0.2) -> Dict:
        """
        Run comprehensive model comparison
        """
        print("=== COMPREHENSIVE MODEL COMPARISON ===")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target range: {y.min():.3f} to {y.max():.3f}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train all models
        rf_results = self.train_random_forest_baseline(X_train, X_test, y_train, y_test)
        nn_results = self.train_neural_network_variants(X_train, X_test, y_train, y_test)
        pinn_results = self.train_physics_informed_nn(X_train, X_test, y_train, y_test, feature_names)
        
        # Compile results
        all_results = {
            'random_forest': rf_results,
            'neural_networks': nn_results,
            'pinn': pinn_results,
            'test_data': {
                'y_true': y_test,
                'X_test': X_test,
                'feature_names': feature_names
            }
        }
        
        # Print summary
        print(f"\n=== FINAL RESULTS SUMMARY ===")
        print(f"Random Forest:     R² = {rf_results['r2']:.4f}")
        
        best_nn_name = max(nn_results.keys(), key=lambda x: nn_results[x]['r2'])
        best_nn_r2 = nn_results[best_nn_name]['r2']
        print(f"Best NN ({best_nn_name:8}): R² = {best_nn_r2:.4f}")
        
        print(f"PINN:              R² = {pinn_results['r2']:.4f}")
        
        # Determine best model
        all_r2_scores = {
            'Random Forest': rf_results['r2'],
            f'NN ({best_nn_name})': best_nn_r2,
            'PINN': pinn_results['r2']
        }
        
        best_model = max(all_r2_scores, key=all_r2_scores.get)
        best_r2 = all_r2_scores[best_model]
        
        print(f"\nBest performing model: {best_model}")
        print(f"Best R² score: {best_r2:.4f}")
        
        if 'PINN' in best_model:
            print("PINN achieves best performance!")
        elif pinn_results['r2'] >= best_r2 - 0.01:
            print("PINN achieves competitive performance!")
        else:
            improvement_needed = best_r2 - pinn_results['r2']
            print(f"PINN needs {improvement_needed:.3f} improvement to match best model")
        
        self.results = all_results
        return all_results

def run_complete_energy_analysis(filepath: str, 
                                sample_size: int = 50000,
                                target_col: str = 'mean') -> Dict:
    """
    Complete energy consumption analysis workflow
    """
    print("Starting Complete Energy Consumption Analysis")
    print("=" * 60)
    
    # Step 1: Data Processing
    processor = LargeDatasetProcessor(sample_size=sample_size)
    
    # Create sample if needed
    try:
        df_sample = pd.read_csv("energy_sample_large.csv")
        print("Using existing sample dataset")
    except FileNotFoundError:
        print("Creating new sample dataset...")
        df_sample = processor.create_sample_dataset(filepath)
    
    # Analyze structure
    structure_analysis = processor.analyze_dataset_structure(df_sample)
    
    # Clean and engineer features
    df_clean = processor.clean_and_engineer_features(df_sample)
    
    # Prepare modeling data
    X, y, feature_names = processor.prepare_modeling_data(df_clean, target_col)
    
    # Step 2: Model Comparison
    comparison = ComprehensiveModelComparison()
    results = comparison.run_comprehensive_comparison(X, y, feature_names)
    
    # Step 3: Visualization
    create_comprehensive_visualizations(results)
    
    return {
        'structure_analysis': structure_analysis,
        'model_results': results,
        'data_info': {
            'original_shape': df_sample.shape,
            'final_shape': (len(X), len(feature_names)),
            'feature_names': feature_names
        }
    }

def create_comprehensive_visualizations(results: Dict):
    """
    Create comprehensive visualization of all results
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Extract data
    rf_results = results['random_forest']
    nn_results = results['neural_networks']
    pinn_results = results['pinn']
    y_true = results['test_data']['y_true']
    feature_names = results['test_data']['feature_names']
    
    # Plot 1: Model performance comparison
    models = ['Random Forest', 'Basic NN', 'Deep NN', 'NN Ensemble', 'PINN']
    r2_scores = [
        rf_results['r2'],
        nn_results['basic']['r2'],
        nn_results['deep']['r2'],
        nn_results['ensemble']['r2'],
        pinn_results['r2']
    ]
    
    colors = ['green', 'blue', 'darkblue', 'lightblue', 'red']
    bars = axes[0,0].bar(models, r2_scores, color=colors, alpha=0.7)
    axes[0,0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('R² Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: PINN vs Random Forest predictions
    axes[0,1].scatter(y_true, pinn_results['predictions'], alpha=0.6, label='PINN', s=20)
    axes[0,1].scatter(y_true, rf_results['predictions'], alpha=0.6, label='Random Forest', s=20)
    axes[0,1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    axes[0,1].set_xlabel('Actual Energy')
    axes[0,1].set_ylabel('Predicted Energy')
    axes[0,1].set_title('PINN vs Random Forest Predictions')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Feature importance from Random Forest
    importance = rf_results['feature_importance']
    top_10_idx = np.argsort(importance)[-10:]
    top_features = [feature_names[i] for i in top_10_idx]
    top_importance = importance[top_10_idx]
    
    axes[0,2].barh(range(len(top_features)), top_importance, alpha=0.8)
    axes[0,2].set_yticks(range(len(top_features)))
    axes[0,2].set_yticklabels(top_features, fontsize=9)
    axes[0,2].set_xlabel('Feature Importance')
    axes[0,2].set_title('Top 10 Features (Random Forest)')
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: PINN residuals
    pinn_residuals = pinn_results['predictions'] - y_true
    axes[1,0].scatter(pinn_results['predictions'], pinn_residuals, alpha=0.6, s=20)
    axes[1,0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1,0].set_xlabel('PINN Predictions')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title('PINN Residual Analysis')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Error distribution comparison
    rf_errors = rf_results['predictions'] - y_true
    pinn_errors = pinn_results['predictions'] - y_true
    
    axes[1,1].hist(rf_errors, bins=30, alpha=0.7, label='Random Forest', color='green')
    axes[1,1].hist(pinn_errors, bins=30, alpha=0.7, label='PINN', color='red')
    axes[1,1].set_xlabel('Prediction Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Error Distribution Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Model metrics comparison
    metrics = ['R²', 'RMSE', 'MAE']
    rf_metrics = [rf_results['r2'], rf_results['rmse'], rf_results['mae']]
    pinn_metrics = [pinn_results['r2'], pinn_results['rmse'], pinn_results['mae']]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    axes[1,2].bar(x_pos - width/2, rf_metrics, width, label='Random Forest', alpha=0.8, color='green')
    axes[1,2].bar(x_pos + width/2, pinn_metrics, width, label='PINN', alpha=0.8, color='red')
    axes[1,2].set_xlabel('Metrics')
    axes[1,2].set_ylabel('Value')
    axes[1,2].set_title('Detailed Metrics Comparison')
    axes[1,2].set_xticks(x_pos)
    axes[1,2].set_xticklabels(metrics)
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    # Plot 7: Neural network architecture comparison
    nn_models = list(nn_results.keys())
    nn_r2_scores = [nn_results[model]['r2'] for model in nn_models]
    
    axes[2,0].bar(nn_models, nn_r2_scores, alpha=0.8, color='lightblue')
    axes[2,0].set_title('Neural Network Variants Comparison')
    axes[2,0].set_ylabel('R² Score')
    axes[2,0].tick_params(axis='x', rotation=45)
    axes[2,0].grid(True, alpha=0.3)
    
    # Plot 8: Best model predictions vs actual
    best_nn_model = max(nn_results.keys(), key=lambda x: nn_results[x]['r2'])
    best_nn_pred = nn_results[best_nn_model]['predictions']
    
    axes[2,1].scatter(y_true, best_nn_pred, alpha=0.6, s=20, color='blue', label=f'Best NN ({best_nn_model})')
    axes[2,1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    axes[2,1].set_xlabel('Actual Energy')
    axes[2,1].set_ylabel('Predicted Energy')
    axes[2,1].set_title(f'Best Neural Network: {best_nn_model}')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    # Plot 9: Summary statistics
    summary_text = f"""
Analysis Summary

Dataset Size: {len(y_true):,} samples
Features: {len(feature_names)}

Model Performance (R²):
Random Forest: {rf_results['r2']:.4f}
Best NN: {max(nn_r2_scores):.4f}
PINN: {pinn_results['r2']:.4f}

Best Model: {models[np.argmax(r2_scores)]}
Performance Range: {min(r2_scores):.4f} - {max(r2_scores):.4f}

Physics-Informed Benefits:
{'+' if pinn_results['r2'] > max(nn_r2_scores) else '-'} vs Neural Networks
{'+' if pinn_results['r2'] > rf_results['r2'] else '-'} vs Random Forest
"""
    
    axes[2,2].text(0.1, 0.5, summary_text, fontsize=10, ha='left', va='center',
                   transform=axes[2,2].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    axes[2,2].set_title('Analysis Summary')
    axes[2,2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nDataset Information:")
    print(f"  Samples: {len(y_true):,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Target range: {y_true.min():.3f} to {y_true.max():.3f}")
    
    print(f"\nModel Performance (R² Score):")
    for model, score in zip(models, r2_scores):
        print(f"  {model:15}: {score:.4f}")
    
    best_model_idx = np.argmax(r2_scores)
    print(f"\nBest Model: {models[best_model_idx]} (R² = {r2_scores[best_model_idx]:.4f})")
    
    if models[best_model_idx] == 'PINN':
        print("SUCCESS: Physics-Informed Neural Network achieves best performance!")
    elif pinn_results['r2'] >= max(r2_scores) - 0.01:
        print("SUCCESS: PINN achieves competitive performance!")
    else:
        print("PINN shows promise but needs optimization for this dataset")

if __name__ == "__main__":
    print("Comprehensive Energy Consumption Analysis Framework")
    print("Supports large datasets with full model comparison")
    print("\nUsage:")
    print("results = run_complete_energy_analysis('your_large_file.csv')")
