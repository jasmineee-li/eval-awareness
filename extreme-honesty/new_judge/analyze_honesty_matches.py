import pandas as pd
import os
from pathlib import Path

# Path to the metrics directory
metrics_dir = Path('/Users/arunim/Documents/github/honest-agents/new_judge/csv_data/metrics')

# Get all CSV files in the directory
csv_files = list(metrics_dir.glob('*.csv'))

# Process each CSV file
for csv_file in csv_files:
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if both columns exist in the dataframe
        if 'true_eval_honesty_1' in df.columns and 'eval_honesty_1' in df.columns:
            # Count matches
            matches = (df['true_eval_honesty_1'] == df['eval_honesty_1']).sum()
            total = len(df)
            
            # Print results
            print(f"File: {csv_file.name}")
            print(f"Matches: {matches}/{total} ({matches/total:.2%})")
            print("-" * 40)
        else:
            print(f"File: {csv_file.name}")
            print("Required columns not found in this file")
            print("-" * 40)
    except Exception as e:
        print(f"Error processing {csv_file.name}: {str(e)}")
        print("-" * 40)
