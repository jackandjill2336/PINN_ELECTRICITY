#!/usr/bin/env python3
"""
Quick PINN test script for rapid prototyping and validation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_synthetic_energy_data(n_samples=5000):
    """Create synthetic energy consumption data for testing"""
    np.random.seed(42)
    
    # Generate realistic weather data
    temp = np.random.normal(12, 8, n_samples)  # Temperature
    hdd = np.maximum(0, 18 - temp) * np.random.uniform(0.8, 1.2, n_samples)  # Heating degree days
    solar = np.random.uniform(0, 8, n_samples)  # Solar radiation
    month = np.random.randint(1, 13, n_samples)  # Month
    
    # Generate energy consumption with physics relationships
    base_energy = 25  # Base consumption
    temp_effect = -0.8 * temp  # Lower temp = higher energy
    hdd_effect = 1.2 * hdd  # Higher HDD = higher energy
    solar_effect = -0.3 * solar  # Higher solar = lower energy
    seasonal_effect = 3 * np.sin(2 * np.pi * month / 12)  # Seasonal variation
    noise = np.random.normal(0, 4, n_samples)  # Random noise
    
    energy = base_energy + temp_effect + hdd_effect + solar_effect + seasonal_effect + noise
    energy = np.maximum(1, energy)  # Ensure positive values
    
    # Create DataFrame
    df = pd.DataFrame({
        'mean_temp': temp,
        'mean_hdd': hdd,
        'mean_solar': solar,
        'month': month,
        'quantity': np.random.choice(['Gas', 'Electricity'], n_samples),
        'unit': np.random.choice(['kWh/day'], n_samples),
        'segment_1_value': np.random.choice(['Type A', 'Type B', 'Type C'], n_samples),
        'mean': energy,
        'aggregation_period': pd.date_range('2021-01-01', periods=n_samples, freq='D')
    })
    
    return df

def run_quick_test():
    """Run quick PINN test with synthetic data"""
    print("Quick PINN Test")
    print("="*50)
    
    # Create synthetic data
    print("Creating synthetic energy data...")
    df = create_synthetic_energy_data(5000)
    df.to_csv("synthetic_energy_data.csv", index=False)
    print(f"   Created {len(df)} samples")
    print(f"   Energy range: {df['mean'].min():.2f} to {df['mean'].max():.2f} kWh")
    
    # Run analysis
    print("\nRunning PINN analysis...")
    try:
        from pinn_electricity.building_energy.comprehensive_analysis import run_complete_energy_analysis
        
        results = run_complete_energy_analysis(
            filepath="synthetic_energy_data.csv",
            sample_size=5000,
            target_col='mean'
        )
        
        # Extract key results
        rf_r2 = results['model_results']['random_forest']['r2']
        pinn_r2 = results['model_results']['pinn']['r2']
        
        print(f"\nTest completed successfully!")
        print(f"Random Forest R²: {rf_r2:.4f}")
        print(f"PINN R²: {pinn_r2:.4f}")
        
        if pinn_r2 > rf_r2:
            print(f"PINN outperforms Random Forest!")
        elif pinn_r2 > 0.85:
            print(f"PINN achieves excellent performance!")
        else:
            print(f"PINN shows reasonable performance")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("   Make sure PINN modules are properly installed")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
