# src/data_cleaning.py
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Perform basic data cleaning operations
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_shape = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    final_shape = df_clean.shape[0]
    
    if initial_shape != final_shape:
        print(f"Removed {initial_shape - final_shape} duplicate rows")
    
    return df_clean

def validate_data(df):
    """
    Validate data quality after cleaning
    """
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.value_counts().to_dict()
    }
    
    return validation_report