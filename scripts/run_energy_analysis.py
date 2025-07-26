scripts/run_energy_analysis.py 

#!/usr/bin/env python3 """ Main script for running comprehensive energy consumption analysis Supports both SERL datasets and custom energy data """ 

import argparse import os import sys import yaml from pathlib import Path 

Add project root to path 

project_root = Path(file).parent.parent sys.path.insert(0, str(project_root)) 

from pinn_electricity.building_energy.comprehensive_analysis import run_complete_energy_analysis 

def main(): parser = argparse.ArgumentParser( description='Run comprehensive energy consumption analysis with PINN', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=""" Examples: python scripts/run_energy_analysis.py --data data.csv python scripts/run_energy_analysis.py --data data.csv --sample-size 30000 python scripts/run_energy_analysis.py --data data.csv --target mean_energy --config custom_config.yaml """ ) 

parser.add_argument('--data', type=str, required=True, 
                  help='Path to energy consumption CSV data file') 
parser.add_argument('--sample-size', type=int, default=50000, 
                  help='Maximum sample size for large datasets (default: 50000)') 
parser.add_argument('--target', type=str, default='mean', 
                  help='Target column name for energy consumption (default: mean)') 
parser.add_argument('--config', type=str,  
                  help='Path to configuration YAML file (optional)') 
parser.add_argument('--output-dir', type=str, default='results', 
                  help='Output directory for results (default: results)') 
parser.add_argument('--no-visualizations', action='store_true', 
                  help='Skip creating visualizations') 
parser.add_argument('--verbose', '-v', action='store_true', 
                  help='Enable verbose output') 
 
args = parser.parse_args() 
 
# Validate input file 
if not os.path.exists(args.data): 
    print(f"Error: Data file '{args.data}' not found") 
    sys.exit(1) 
 
# Create output directory 
output_dir = Path(args.output_dir) 
output_dir.mkdir(exist_ok=True) 
 
# Load configuration if provided 
config = {} 
if args.config: 
    if not os.path.exists(args.config): 
        print(f"Warning: Config file '{args.config}' not found, using defaults") 
    else: 
        with open(args.config, 'r') as f: 
            config = yaml.safe_load(f) 
 
print("="*60) 
print("COMPREHENSIVE ENERGY CONSUMPTION ANALYSIS") 
print("="*60) 
print(f"Data file: {args.data}") 
print(f"Sample size: {args.sample_size:,}") 
print(f"Target column: {args.target}") 
print(f"Output directory: {output_dir}") 
print(f"Configuration: {'Custom' if args.config else 'Default'}") 
print("="*60) 
 
try: 
    # Run analysis 
    results = run_complete_energy_analysis( 
        filepath=args.data, 
        sample_size=args.sample_size, 
        target_col=args.target 
    ) 
     
    # Save results 
    import pickle 
    import json 
     
    results_file = output_dir / 'analysis_results.pkl' 
    with open(results_file, 'wb') as f: 
        pickle.dump(results, f) 
     
    # Save summary as JSON 
    summary = { 
        'data_info': results['data_info'], 
        'model_performance': { 
            'random_forest_r2': results['model_results']['random_forest']['r2'], 
            'pinn_r2': results['model_results']['pinn']['r2'], 
            'best_nn_r2': max([results['model_results']['neural_networks'][model]['r2']  
                             for model in results['model_results']['neural_networks']]), 
        } 
    } 
     
    summary_file = output_dir / 'analysis_summary.json' 
    with open(summary_file, 'w') as f: 
        json.dump(summary, f, indent=2) 
     
    print(f"\n✅ Analysis completed successfully!") 
    print(f"📊 Results saved to: {results_file}") 
    print(f"📋 Summary saved to: {summary_file}") 
     
    # Print key results 
    rf_r2 = results['model_results']['random_forest']['r2'] 
    pinn_r2 = results['model_results']['pinn']['r2'] 
     
    print(f"\n🎯 Key Results:") 
    print(f"   Random Forest R²: {rf_r2:.4f}") 
    print(f"   PINN R²: {pinn_r2:.4f}") 
     
    if pinn_r2 > rf_r2: 
        improvement = ((pinn_r2 - rf_r2) / rf_r2) * 100 
        print(f"   🎉 PINN outperforms baseline by {improvement:.1f}%!") 
    else: 
        print(f"   📈 PINN achieves competitive performance") 
     
except Exception as e: 
    print(f"\n❌ Analysis failed with error: {e}") 
    if args.verbose: 
        import traceback 
        traceback.print_exc() 
    sys.exit(1) 
  

if name == "main": main() 

scripts/quick_pinn_test.py 

#!/usr/bin/env python3 """ Quick PINN test script for rapid prototyping and validation """ 

import sys from pathlib import Path import numpy as np import pandas as pd 

Add project root to path 

project_root = Path(file).parent.parent sys.path.insert(0, str(project_root)) 

def create_synthetic_energy_data(n_samples=5000): """Create synthetic energy consumption data for testing""" np.random.seed(42) 

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
  

def run_quick_test(): """Run quick PINN test with synthetic data""" print("🚀 Quick PINN Test") print("="*50) 

# Create synthetic data 
print("📊 Creating synthetic energy data...") 
df = create_synthetic_energy_data(5000) 
df.to_csv("synthetic_energy_data.csv", index=False) 
print(f"   Created {len(df)} samples") 
print(f"   Energy range: {df['mean'].min():.2f} to {df['mean'].max():.2f} kWh") 
 
# Run analysis 
print("\n🧠 Running PINN analysis...") 
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
     
    print(f"\n✅ Test completed successfully!") 
    print(f"📈 Random Forest R²: {rf_r2:.4f}") 
    print(f"🧠 PINN R²: {pinn_r2:.4f}") 
     
    if pinn_r2 > rf_r2: 
        print(f"🎉 PINN outperforms Random Forest!") 
    elif pinn_r2 > 0.85: 
        print(f"✨ PINN achieves excellent performance!") 
    else: 
        print(f"📊 PINN shows reasonable performance") 
     
    return True 
     
except ImportError as e: 
    print(f"❌ Import error: {e}") 
    print("   Make sure PINN modules are properly installed") 
    return False 
except Exception as e: 
    print(f"❌ Test failed: {e}") 
    return False 
  

if name == "main": success = run_quick_test() sys.exit(0 if success else 1) 

configs/comprehensive_config.yaml 

Comprehensive Configuration for PINN Energy Analysis 

Dataset configuration 

dataset: sample_size: 50000 target_column: "mean" test_size: 0.2 random_state: 42 

Data cleaning parameters 

cleaning: min_energy_threshold: 0.001 outlier_method: "iqr" outlier_factor: 1.5 

Feature engineering 

features: use_cyclical_encoding: true use_physics_interactions: true use_temperature_zones: true use_seasonal_features: true 

# Physics features to create 
physics_features: 
  - "temp_hdd_interaction" 
  - "temp_squared"  
  - "hdd_squared" 
  - "solar_temp_interaction" 
   
# Temperature zones 
temperature_zones: 
  very_cold: 5 
  cold: 15 
  mild: 20 
  

Model configurations 

models: random_forest: n_estimators: 200 random_state: 42 n_jobs: -1 

neural_networks: basic: hidden_layers: [128, 64, 32] activation: "relu" epochs: 100 batch_size: 32 learning_rate: 0.001 

deep: 
  hidden_layers: [256, 128, 64, 32, 16] 
  activation: "relu" 
  dropout_rates: [0.1, 0.1, 0.0, 0.0, 0.0] 
  epochs: 150 
  batch_size: 32 
  learning_rate: 0.001 
   
ensemble: 
  n_models: 3 
  hidden_layers: [128, 64, 32] 
  activation: "relu" 
  epochs: 100 
  batch_size: 32 
  learning_rate: 0.001 
   
  

pinn: # Physics-aware architecture branches: temperature: hidden_layers: [16, 8] activation: "tanh" heating: hidden_layers: [16, 8] activation: "relu" solar: hidden_layers: [12, 6] activation: "sigmoid" seasonal: hidden_layers: [8] activation: "tanh" 

# Main processing network 
main_network: 
  hidden_layers: [64, 32, 16] 
  activation: "relu" 
  dropout_rates: [0.2, 0.1, 0.0] 
  use_batch_norm: true 
   
# Training parameters 
training: 
  epochs: 100 
  batch_size: 64 
  learning_rate: 0.001 
  physics_weight: 0.1 
   
# Physics constraints 
physics: 
  enable_hdd_correlation: true 
  enable_temperature_zones: true 
  enable_energy_bounds: true 
  enable_efficiency_consistency: true 
  

Output configuration 

output: save_models: true save_predictions: true save_visualizations: true create_report: true 

Visualization options 

visualizations: model_comparison: true feature_importance: true residual_analysis: true error_distributions: true physics_analysis: true 

Report options 

report: include_methodology: true include_results_summary: true include_recommendations: true format: "html" # html, pdf, or markdown 

Advanced options 

advanced: enable_gradcam: false # Set to true for interpretability analysis gradcam_samples: 30 enable_hyperparameter_tuning: false cross_validation_folds: 5 

Logging 

logging: level: "INFO" # DEBUG, INFO, WARNING, ERROR save_logs: true log_file: "analysis.log" 

Example notebook configurations/example_config.yaml 

Example Configuration for Different Use Cases 

Quick test configuration 

quick_test: dataset: sample_size: 5000 target_column: "mean" models: neural_networks: basic: epochs: 50 pinn: training: epochs: 50 

High accuracy configuration 

high_accuracy: dataset: sample_size: 100000 models: random_forest: n_estimators: 500 neural_networks: deep: epochs: 300 hidden_layers: [512, 256, 128, 64, 32] pinn: training: epochs: 200 

Research configuration 

research: advanced: enable_gradcam: true enable_hyperparameter_tuning: true cross_validation_folds: 10 output: create_report: true save_visualizations: true 

examples/notebooks/comprehensive_energy_analysis.py 

""" Jupyter Notebook: Comprehensive Energy Analysis with PINN Convert this to .ipynb for interactive use """ 

Cell 1: Setup and Imports 

import sys sys.path.append('../../') 

import numpy as np import pandas as pd import matplotlib.pyplot as plt from pinn_electricity.building_energy.comprehensive_analysis import run_complete_energy_analysis 

Cell 2: Load Configuration 

import yaml 

Load default configuration 

with open('../../configs/comprehensive_config.yaml', 'r') as f: config = yaml.safe_load(f) 

print("Configuration loaded:") print(f"Sample size: {config['dataset']['sample_size']:,}") print(f"Target column: {config['dataset']['target_column']}") 

Cell 3: Data Analysis 

You can use either: 

1. Your actual SERL data file 

2. The synthetic data generator for testing 

For actual data: 

filepath = "path/to/your/serl_data.csv" 

For testing with synthetic data: 

print("Creating synthetic test data...") exec(open('../../scripts/quick_pinn_test.py').read()) 

Cell 4: Run Comprehensive Analysis 

print("Running comprehensive energy consumption analysis...") 

results = run_complete_energy_analysis( filepath="synthetic_energy_data.csv", # Change to your actual file sample_size=config['dataset']['sample_size'], target_col=config['dataset']['target_column'] ) 

Cell 5: Results Analysis 

model_results = results['model_results'] 

print("=== ANALYSIS RESULTS ===") print(f"Random Forest R²: {model_results['random_forest']['r2']:.4f}") print(f"PINN R²: {model_results['pinn']['r2']:.4f}") 

Neural network results 

nn_results = model_results['neural_networks'] for model_name, result in nn_results.items(): print(f"{model_name} NN R²: {result['r2']:.4f}") 

Cell 6: Custom Visualization 

Create your own custom plots here 

y_true = model_results['test_data']['y_true'] pinn_pred = model_results['pinn']['predictions'] rf_pred = model_results['random_forest']['predictions'] 

plt.figure(figsize=(12, 5)) 

plt.subplot(1, 2, 1) plt.scatter(y_true, pinn_pred, alpha=0.6, label='PINN') plt.scatter(y_true, rf_pred, alpha=0.6, label='Random Forest') plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--') plt.xlabel('Actual Energy') plt.ylabel('Predicted Energy') plt.legend() plt.title('Model Predictions Comparison') 

plt.subplot(1, 2, 2) pinn_error = pinn_pred - y_true rf_error = rf_pred - y_true plt.hist(pinn_error, alpha=0.7, label='PINN Error', bins=30) plt.hist(rf_error, alpha=0.7, label='RF Error', bins=30) plt.xlabel('Prediction Error') plt.ylabel('Frequency') plt.legend() plt.title('Error Distribution') 

plt.tight_layout() plt.show() 

Cell 7: Physics Analysis 

Analyze if PINN learned physics relationships 

feature_names = model_results['test_data']['feature_names'] rf_importance = model_results['random_forest']['feature_importance'] 

print("=== PHYSICS FEATURE ANALYSIS ===") physics_features = ['temp_hdd_interaction', 'temp_squared', 'hdd_squared'] for i, feature in enumerate(feature_names): if feature in physics_features: print(f"{feature}: Importance = {rf_importance[i]:.4f}") 

Cell 8: Save Results 

import pickle import json from datetime import datetime 

Save complete results 

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") results_file = f"analysis_results_{timestamp}.pkl" 

with open(results_file, 'wb') as f: pickle.dump(results, f) 

print(f"Results saved to: {results_file}") 

Create summary report 

summary = { 'timestamp': timestamp, 'dataset_info': results['data_info'], 'model_performance': { 'random_forest_r2': model_results['random_forest']['r2'], 'pinn_r2': model_results['pinn']['r2'], 'best_nn_r2': max([nn_results[model]['r2'] for model in nn_results]) }, 'physics_analysis': { 'pinn_beats_rf': model_results['pinn']['r2'] > model_results['random_forest']['r2'], 'improvement': model_results['pinn']['r2'] - model_results['random_forest']['r2'] } } 

summary_file = f"analysis_summary_{timestamp}.json" with open(summary_file, 'w') as f: json.dump(summary, f, indent=2) 

print(f"Summary saved to: {summary_file}") print("\n✅ Comprehensive analysis complete!") 

 
