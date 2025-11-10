# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_target_distribution(df, target_column='Target'):
    """
    Plot distribution of target variable
    """
    if target_column in df.columns:
        plt.figure(figsize=(10, 6))
        df[target_column].value_counts().plot(kind='bar')
        plt.title('Distribution of Target Variable')
        plt.xlabel('Target Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def plot_correlation_heatmap(df, figsize=(12, 10)):
    """
    Plot correlation heatmap for numerical features
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 1:
        plt.figure(figsize=figsize)
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()