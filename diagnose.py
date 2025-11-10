# diagnose.py
import pandas as pd
import numpy as np
import os
import sys

def quick_diagnose():
    """Quick diagnosis of the dataset"""
    try:
        # Check raw data
        print("📁 CHECKING RAW DATA...")
        with open('data/data.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"File has {len(lines)} lines")
        print("First 3 lines:")
        for i, line in enumerate(lines[:3]):
            print(f"Line {i}: {repr(line)}")
        
        # Try to load raw data
        try:
            raw_df = pd.read_csv('data/data.csv')
            print(f"Raw data shape: {raw_df.shape}")
            print(f"Raw data columns: {list(raw_df.columns)}")
        except Exception as e:
            print(f"❌ Cannot load raw data: {e}")
        
        # Check cleaned data
        print("\n📁 CHECKING CLEANED DATA...")
        if os.path.exists('data/cleaned_student_data.csv'):
            df = pd.read_csv('data/cleaned_student_data.csv')
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            if 'Target' in df.columns:
                print(f"\n🎯 TARGET ANALYSIS:")
                print(f"Unique values: {df['Target'].unique()}")
                print(f"Distribution:\n{df['Target'].value_counts()}")
                print(f"Proportions:\n{df['Target'].value_counts(normalize=True)}")
            
            # Check for issues
            print(f"\n🔍 DATA ISSUES:")
            print(f"Missing values: {df.isnull().sum().sum()}")
            print(f"Duplicate rows: {df.duplicated().sum()}")
            
            constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
            if constant_cols:
                print(f"Constant columns: {constant_cols}")
            
            # Check data types
            print(f"\n📊 DATA TYPES:")
            print(df.dtypes.value_counts())
            
        else:
            print("❌ Cleaned data file not found!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_diagnose()