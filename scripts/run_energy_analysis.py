#!/usr/bin/env python3
"""
Main script for running comprehensive energy consumption analysis
Supports both SERL datasets and custom energy data
"""

import argparse
import os
import sys
import yaml
import json
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinn_electricity.building_energy.comprehensive_analysis import run_complete_energy_analysis

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive energy consumption analysis with PINN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_energy_analysis.py --data data.csv
  python scripts/run_energy_analysis.py --data data.csv --sample-size 30000
  python scripts/run_energy_analysis.py --data data.csv --target mean_energy --config custom_config.yaml
        """
    )
    
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
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        
        # Print key results
        rf_r2 = results['model_results']['random_forest']['r2']
        pinn_r2 = results['model_results']['pinn']['r2']
        
        print(f"\nKey Results:")
        print(f"   Random Forest R²: {rf_r2:.4f}")
        print(f"   PINN R²: {pinn_r2:.4f}")
        
        if pinn_r2 > rf_r2:
            improvement = ((pinn_r2 - rf_r2) / rf_r2) * 100
            print(f"   PINN outperforms baseline by {improvement:.1f}%!")
        else:
            print(f"   PINN achieves competitive performance")
        
    except Exception as e:
        print(f"\nAnalysis failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
