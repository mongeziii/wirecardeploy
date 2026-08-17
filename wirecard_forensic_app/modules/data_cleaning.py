"""Data cleaning utilities."""
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """Basic cleaning operations."""
    df = df.copy()
    # Remove duplicate rows
    df = df.drop_duplicates()
    # Fill numeric NaNs with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df
