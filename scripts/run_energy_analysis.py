#!/usr/bin/env python3
"""
Main script for running comprehensive energy consumption analysis
Supports both SERL datasets and custom energy data
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinn_electricity.building_energy.comprehensive_analysis import run_complete_energy_analysis

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive energy consumption analysis with PINN'
    )

    parser.add_argument('--data', type=str, required=True,
                       help='Path to energy consumption CSV data file')
    parser.add_argument('--sample-size', type=int, default=50000,
                       help='Maximum sample size for large datasets')
    parser.add_argument('--target', type=str, default='mean',
                       help='Target column name for energy consumption')

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found")
        sys.exit(1)

    print("COMPREHENSIVE ENERGY CONSUMPTION ANALYSIS")
    print(f"Data file: {args.data}")
    print(f"Sample size: {args.sample_size:,}")

    try:
        results = run_complete_energy_analysis(
            filepath=args.data,
            sample_size=args.sample_size,
            target_col=args.target
        )

        rf_r2 = results['model_results']['random_forest']['r2']
        pinn_r2 = results['model_results']['pinn']['r2']

        print(f"Random Forest R²: {rf_r2:.4f}")
        print(f"PINN R²: {pinn_r2:.4f}")

        if pinn_r2 > rf_r2:
            improvement = ((pinn_r2 - rf_r2) / rf_r2) * 100
            print(f"PINN outperforms baseline by {improvement:.1f}%")

    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
