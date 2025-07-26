# pinn_electricity/building_energy/feature_engineering.py
"""
Advanced Feature Engineering and Model Architecture
Based on proven approaches for building energy consumption prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for building energy consumption
    Implements all proven feature engineering approaches
    """
    
    def __init__(self):
        self.feature_importance_scores = {}
        self.polynomial_features = None
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables
        Based on proven approach: temp_hdd_interaction was most successful
        """
        df_interactions = df.copy()
        
        # Primary interaction (most successful)
        df_interactions['temp_hdd_interaction'] = df_interactions['mean_temp'] * df_interactions['mean_hdd']
        
        # Additional interactions
        if 'building_segment' in df_interactions.columns:
            df_interactions['building_hdd'] = df_interactions['building_segment'] * df_interactions['mean_hdd']
            df_interactions['building_temp'] = df_interactions['building_segment'] * df_interactions['mean_temp']
        
        if 'month' in df_interactions.columns:
            df_interactions['temp_month'] = df_interactions['mean_temp'] * df_interactions['month']
            df_interactions['hdd_month'] = df_interactions['mean_hdd'] * df_interactions['month']
            
        # Higher order interactions
        df_interactions['temp_hdd_squared'] = df_interactions['temp_hdd_interaction'] ** 2
        
        print("Created interaction features")
        return df_interactions
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        Based on testing: degree 2 showed best performance
        """
        df_poly = df.copy()
        
        # Manual polynomial features (more control than sklearn)
        df_poly['temp_squared'] = df_poly['mean_temp'] ** 2
        df_poly['hdd_squared'] = df_poly['mean_hdd'] ** 2
        
        if degree >= 3:
            df_poly['temp_cubed'] = df_poly['mean_temp'] ** 3
            df_poly['hdd_cubed'] = df_poly['mean_hdd'] ** 3
            
        if 'building_segment' in df_poly.columns:
            df_poly['building_squared'] = df_poly['building_segment'] ** 2
            
        print(f"Created polynomial features (degree {degree})")
        return df_poly
    
    def create_temperature_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temperature threshold features
        Based on proven approach: temp_below_10, temp_below_5, temp_below_0
        """
        df_thresholds = df.copy()
        
        # Your proven temperature thresholds
        df_thresholds['temp_below_10'] = (df_thresholds['mean_temp'] < 10).astype(float)
        df_thresholds['temp_below_5'] = (df_thresholds['mean_temp'] < 5).astype(float)
        df_thresholds['temp_below_0'] = (df_thresholds['mean_temp'] < 0).astype(float)
        
        # Additional temperature zones
        df_thresholds['temp_above_20'] = (df_thresholds['mean_temp'] > 20).astype(float)
        df_thresholds['temp_above_15'] = (df_thresholds['mean_temp'] > 15).astype(float)
        
        # Temperature ranges
        df_thresholds['very_cold'] = (df_thresholds['mean_temp'] < 5).astype(float)
        df_thresholds['cold'] = ((df_thresholds['mean_temp'] >= 5) & 
                                (df_thresholds['mean_temp'] < 15)).astype(float)
        df_thresholds['mild'] = ((df_thresholds['mean_temp'] >= 15) & 
                                (df_thresholds['mean_temp'] < 20)).astype(float)
        df_thresholds['warm'] = (df_thresholds['mean_temp'] >= 20).astype(float)
        
        print("Created temperature threshold features")
        return df_thresholds
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal and time-based features
        Based on proven approach: is_winter, is_heating_season
        """
        df_seasonal = df.copy()
        
        if 'month' in df_seasonal.columns:
            # Your proven seasonal features
            df_seasonal['is_winter'] = df_seasonal['month'].isin([12, 1, 2]).astype(float)
            df_seasonal['is_heating_season'] = df_seasonal['month'].isin([10, 11, 12, 1, 2, 3]).astype(float)
            
            # Additional seasonal features
            df_seasonal['is_summer'] = df_seasonal['month'].isin([6, 7, 8]).astype(float)
            df_seasonal['is_shoulder_season'] = df_seasonal['month'].isin([3, 4, 9, 10]).astype(float)
            
            # Winter heating intensity
            df_seasonal['winter_heating'] = ((df_seasonal['month'].isin([12, 1, 2])) & 
                                           (df_seasonal['mean_temp'] < 15)).astype(float)
            
        print("Created seasonal features")
        return df_seasonal
    
    def create_physics_informed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create physics-informed features based on building thermodynamics
        """
        df_physics = df.copy()
        
        # Heating demand features (nonlinear relationship)
        df_physics['hdd_sqrt'] = np.sqrt(np.maximum(0, df_physics['mean_hdd']))
        df_physics['hdd_cuberoot'] = np.cbrt(np.maximum(0, df_physics['mean_hdd']))
        
        # Effective heating demand
        df_physics['hdd_effective'] = np.maximum(0, df_physics['mean_hdd']) ** 0.8
        
        # Thermal comfort deviation
        comfort_temp = 20
        df_physics['temp_deviation'] = np.abs(df_physics['mean_temp'] - comfort_temp)
        df_physics['temp_deviation_squared'] = df_physics['temp_deviation'] ** 2
        
        # Heating efficiency proxies
        df_physics['heating_efficiency'] = df_physics['value_kWh'] / (df_physics['mean_hdd'] + 1e-6)
        
        # Temperature efficiency factor
        df_physics['temp_efficiency'] = np.where(df_physics['mean_temp'] < 15,
                                               1.0 + 0.1 * (15 - df_physics['mean_temp']),
                                               1.0)
        
        # Building thermal mass proxy
        if 'building_segment' in df_physics.columns:
            df_physics['thermal_mass'] = df_physics['building_segment'] * 0.1
            
        print("Created physics-informed features")
        return df_physics
    
    def create_all_features(self, df: pd.DataFrame, 
                          include_interactions: bool = True,
                          include_polynomials: bool = True,
                          include_thresholds: bool = True,
                          include_seasonal: bool = True,
                          include_physics: bool = True,
                          polynomial_degree: int = 2) -> pd.DataFrame:
        """
        Create all engineered features
        """
        df_features = df.copy()
        
        if include_interactions:
            df_features = self.create_interaction_features(df_features)
            
        if include_polynomials:
            df_features = self.create_polynomial_features(df_features, degree=polynomial_degree)
            
        if include_thresholds:
            df_features = self.create_temperature_thresholds(df_features)
            
        if include_seasonal:
            df_features = self.create_seasonal_features(df_features)
            
        if include_physics:
            df_features = self.create_physics_informed_features(df_features)
            
        print(f"Created comprehensive feature set: {df_features.shape[1]} total features")
        return df_features
    
    def select_best_features(self, df: pd.DataFrame, 
                           target_col: str = 'value_kWh',
                           method: str = 'random_forest',
                           max_features: int = 20) -> List[str]:
        """
        Select best features using specified method
        """
        # Start with core features
        core_features = ['mean_temp', 'mean_hdd', 'month']
        if 'building_segment' in df.columns:
            core_features.append('building_segment')
            
        # Get all potential features (excluding target)
        all_features = [col for col in df.columns if col != target_col and not col.startswith('date')]
        
        if method == 'random_forest':
            # Use Random Forest feature importance
            X = df[all_features].fillna(0)
            y = df[target_col]
            
            # Remove any rows with missing target
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': all_features,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = importance_df.head(max_features)['feature'].tolist()
            
            # Ensure core features are included
            for core_feat in core_features:
                if core_feat in all_features and core_feat not in selected_features:
                    selected_features.append(core_feat)
                    
            self.feature_importance_scores = dict(zip(importance_df['feature'], importance_df['importance']))
            
        else:
            # Default to core features + most common engineered features
            selected_features = core_features.copy()
            
            # Add most successful engineered features
            priority_features = [
                'temp_hdd_interaction', 'temp_squared', 'hdd_squared',
                'temp_below_10', 'temp_below_5', 'is_winter', 'is_heating_season'
            ]
            
            for feat in priority_features:
                if feat in all_features and feat not in selected_features:
                    selected_features.append(feat)
                    
        print(f"Selected {len(selected_features)} features")
        return selected_features

class ModelArchitectures:
    """
    Neural network model architectures for energy prediction
    Based on proven approaches and experimental results
    """
    
    @staticmethod
    def create_simple_nn(input_dim: int, 
                        hidden_layers: List[int] = [32, 16],
                        activation: str = 'relu',
                        dropout_rate: float = 0.0) -> keras.Model:
        """
        Create simple neural network (baseline approach)
        """
        model = keras.Sequential()
        model.add(keras.Input(shape=(input_dim,)))
        
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation=activation, name=f'dense_{i+1}'))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))
                
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        return model
    
    @staticmethod
    def create_multi_pathway_nn(input_dim: int) -> keras.Model:
        """
        Create multi-pathway neural network
        Based on WorkingPINN approach with separate pathways for different feature types
        """
        inputs = keras.Input(shape=(input_dim,), name='inputs')
        
        # Temperature pathway
        temp_features = layers.Dense(16, activation='swish', name='temp_pathway')(inputs)
        temp_features = layers.Dense(8, activation='swish')(temp_features)
        
        # HDD pathway
        hdd_features = layers.Dense(16, activation='relu', name='hdd_pathway')(inputs)
        hdd_features = layers.Dense(8, activation='relu')(hdd_features)
        
        # Building pathway
        building_features = layers.Dense(8, activation='relu', name='building_pathway')(inputs)
        
        # Seasonal pathway
        seasonal_features = layers.Dense(8, activation='tanh', name='seasonal_pathway')(inputs)
        
        # Combine pathways
        combined = layers.Concatenate(name='pathway_combination')([
            temp_features, hdd_features, building_features, seasonal_features
        ])
        
        # Main processing
        x = layers.Dense(32, activation='swish', name='main_1')(combined)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation='swish', name='main_2')(x)
        x = layers.Dropout(0.05)(x)
        
        # Output
        output = layers.Dense(1, activation='linear', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=output, name='MultiPathwayNN')
        return model
    
    @staticmethod
    def create_attention_nn(input_dim: int) -> keras.Model:
        """
        Create neural network with attention mechanism
        """
        inputs = keras.Input(shape=(input_dim,), name='inputs')
        
        # Feature extraction
        x = layers.Dense(64, activation='swish', name='feature_extraction')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        attention_weights = layers.Dense(64, activation='sigmoid', name='attention')(x)
        attended_features = layers.Multiply(name='attended_features')([x, attention_weights])
        
        # Main network
        x = layers.Dense(32, activation='swish', name='main_1')(attended_features)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation='swish', name='main_2')(x)
        
        # Output
        output = layers.Dense(1, activation='linear', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=output, name='AttentionNN')
        return model

class ModelExperiments:
    """
    Class for running model experiments and comparisons
    Based on proven experimental approaches
    """
    
    def __init__(self):
        self.results = {}
        
    def test_activation_functions(self, X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray,
                                activations: List[str] = ['relu', 'tanh', 'swish', 'elu'],
                                epochs: int = 100) -> Dict[str, float]:
        """
        Test different activation functions
        Based on proven experimental approach
        """
        print("Testing activation functions...")
        activation_results = {}
        
        for activation in activations:
            print(f"Testing {activation}...")
            
            model = ModelArchitectures.create_simple_nn(
                input_dim=X_train.shape[1],
                hidden_layers=[64, 32],
                activation=activation,
                dropout_rate=0.1
            )
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0,
                     validation_split=0.2)
            
            y_pred = model.predict(X_test, verbose=0).flatten()
            r2 = r2_score(y_test, y_pred)
            activation_results[activation] = r2
            
            print(f"{activation}: R² = {r2:.3f}")
            
        return activation_results
    
    def test_polynomial_features(self, X: pd.DataFrame, y: pd.Series,
                                degrees: List[int] = [2, 3],
                                test_size: float = 0.2) -> Dict[int, float]:
        """
        Test polynomial feature degrees
        Based on proven experimental approach
        """
        print("Testing polynomial features...")
        poly_results = {}
        
        for degree in degrees:
            print(f"Testing degree {degree}...")
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            
            print(f"Degree {degree}: {X_poly.shape[1]} features")
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_poly, y, test_size=test_size, random_state=42
            )
            
            # Simple model for polynomial features
            model = ModelArchitectures.create_simple_nn(
                input_dim=X_train.shape[1],
                hidden_layers=[128, 64, 32],
                activation='relu',
                dropout_rate=0.2
            )
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
            
            y_pred = model.predict(X_test, verbose=0).flatten()
            r2 = r2_score(y_test, y_pred)
            poly_results[degree] = r2
            
            print(f"Polynomial degree {degree}: R² = {r2:.3f}")
            
        return poly_results
    
    def compare_architectures(self, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray,
                            epochs: int = 100) -> Dict[str, float]:
        """
        Compare different neural network architectures
        """
        print("Comparing neural network architectures...")
        
        architectures = {
            'Simple NN': ModelArchitectures.create_simple_nn(X_train.shape[1]),
            'Multi-pathway NN': ModelArchitectures.create_multi_pathway_nn(X_train.shape[1]),
            'Attention NN': ModelArchitectures.create_attention_nn(X_train.shape[1])
        }
        
        architecture_results = {}
        
        for name, model in architectures.items():
            print(f"Testing {name}...")
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0,
                     validation_split=0.2)
            
            y_pred = model.predict(X_test, verbose=0).flatten()
            r2 = r2_score(y_test, y_pred)
            architecture_results[name] = r2
            
            print(f"{name}: R² = {r2:.3f}")
            
        return architecture_results
    
    def run_comprehensive_experiments(self, df: pd.DataFrame,
                                    target_col: str = 'value_kWh',
                                    test_size: float = 0.2) -> Dict:
        """
        Run comprehensive model experiments
        """
        print("Running comprehensive experiments...")
        
        # Feature engineering
        engineer = AdvancedFeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        # Feature selection
        selected_features = engineer.select_best_features(df_features, target_col)
        
        # Prepare data
        X = df_features[selected_features].fillna(0)
        y = df_features[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale data
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Run experiments
        results = {
            'selected_features': selected_features,
            'feature_importance': engineer.feature_importance_scores,
            'activation_results': self.test_activation_functions(X_train_scaled, X_test_scaled, 
                                                               y_train_scaled, y_test_scaled),
            'architecture_results': self.compare_architectures(X_train_scaled, X_test_scaled,
                                                             y_train_scaled, y_test_scaled),
            'polynomial_results': self.test_polynomial_features(X, y)
        }
        
        return results

if __name__ == "__main__":
    print("Advanced Feature Engineering and Model Architecture")
    print("Example usage:")
    print("engineer = AdvancedFeatureEngineer()")
    print("df_features = engineer.create_all_features(df)")
    print("selected_features = engineer.select_best_features(df_features)")
    print("")
    print("experiments = ModelExperiments()")
    print("results = experiments.run_comprehensive_experiments(df)")
