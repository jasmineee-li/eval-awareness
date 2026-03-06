#!/usr/bin/env python3
"""
clean.py - Script to clean CSV files by removing specified columns.

This script processes all CSV files in the responses directory and removes columns
that match certain patterns, then saves the cleaned files to a new 'cleaned' directory.
"""

import os
import glob
import pandas as pd
import re
from pathlib import Path

def clean_csv_file(input_file, output_file):
    """
    Clean a CSV file by removing specified columns.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the cleaned CSV file
    
    Returns:
        pd.DataFrame: The cleaned DataFrame
    """
    print(f"Processing: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get the original column count
    original_columns = len(df.columns)
    
    # Columns to drop
    columns_to_drop = []
    
    for col in df.columns:
        # Exact column matches
        if col in ["ground_truth_eval", "ground_truth_extracted", 
                  "system_user_eval", "system_user_extracted"]:
            columns_to_drop.append(col)
            continue
        
        # Pattern matches
        if (col.startswith("belief_eval_") or 
            col.startswith("eval(") or 
            col.startswith("eval_") or
            "eval(" in col or
            "_eval_" in col or
            col.startswith("human_eval")):
            columns_to_drop.append(col)
            continue
        
        # Generation columns with run2 through run10
        if col.startswith("generation") and re.search(r"_run([2-9]|10)$", col):
            columns_to_drop.append(col)
            continue
    
    # Drop the columns
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Get the new column count
    new_columns = len(df.columns)
    dropped_count = original_columns - new_columns
    
    print(f"Dropped {dropped_count} columns. Original: {original_columns}, New: {new_columns}")
    
    # Save the cleaned DataFrame
    df.to_csv(output_file, index=False)
    print(f"Saved cleaned file to: {output_file}")
    
    return df

def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths
    responses_dir = os.path.join(script_dir, "responses")
    cleaned_dir = os.path.join(script_dir, "cleaned")
    
    # Create the cleaned directory if it doesn't exist
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Get all CSV files in the responses directory
    csv_files = glob.glob(os.path.join(responses_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {responses_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for input_file in csv_files:
        # Create output file path
        filename = os.path.basename(input_file)
        output_file = os.path.join(cleaned_dir, filename)
        
        # Clean the CSV file
        clean_csv_file(input_file, output_file)
    
    print("All files processed successfully!")

if __name__ == "__main__":
    main()
