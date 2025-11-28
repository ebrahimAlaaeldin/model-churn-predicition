import pandas as pd

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def convert_to_numeric(df, column, fill_method=None):
    """
    Convert a DataFrame column to numeric, coercing errors to NaN.
    """

    df = df.copy()
    
    # Convert to numeric (invalid values → NaN)
    df[column] = pd.to_numeric(df[column], errors='coerce')

    # Handle missing values if requested
    if fill_method == "median":
        df[column].fillna(df[column].median(), inplace=True)
    elif fill_method == "mean":
        df[column].fillna(df[column].mean(), inplace=True)
    elif fill_method == "zero":
        df[column].fillna(0, inplace=True)

    return df


def drop_column(df, column_name):
    """
    Safely drop a column from a DataFrame if it exists.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    column_name : str
        Column name to drop.

    Returns
    -------
    df : pandas.DataFrame
        Updated DataFrame with the column removed (if present).
    """
    df = df.copy()

    if column_name in df.columns:
        df.drop(column_name, axis=1, inplace=True)
    else:
        print(f"⚠️ Column '{column_name}' not found — nothing dropped.")

    return df


def encode_categorical_features(df):
    """
    Encodes categorical features using:
      - Label Encoding for binary (Yes/No) columns
      - One-Hot Encoding for multi-category columns
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    df_processed : pandas.DataFrame
        DataFrame with encoded categorical features.
    label_encoders : dict
        Dictionary containing LabelEncoders for binary columns.
    """
    
    df_processed = df.copy()

    # -------- 1️⃣ Label Encode Binary Columns --------
    binary_cols = df_processed.select_dtypes(include=['object']).columns
    binary_cols = [col for col in binary_cols if df_processed[col].nunique() == 2]

    label_encoders = {}

    for col in binary_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    
    # -------- 2️⃣ One-Hot Encode Multi-category Columns --------
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        
    
    df_processed.replace({True:1, False:0})

    return df_processed, label_encoders



def encode_categorical_features_api(df):
    """
    One-hot encode ALL categorical columns without dropping any category.
    """
    df_processed = df.copy()

    categorical_cols = df_processed.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False)
        

    return df_processed, None

## Outliers

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_outliers_boxplot(df, columns=None):
    """
    Visualizes outliers using boxplots for the selected numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    columns : list or None
        Columns to visualize. If None, auto-detect numeric columns.
    """

    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns

    plt.figure(figsize=(15, 5))

    for i, col in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.boxplot(y=df[col], color='skyblue')
        plt.title(col)

    plt.tight_layout()
    plt.show()
    
    
def visualize_outliers_hist(df, columns=None):
    """
    Visualizes distribution of numeric features to spot outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame.
    columns : list or None
        Columns to visualize. If None, detect numeric columns.
    """

    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns

    plt.figure(figsize=(15, 5))

    for i, col in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.histplot(df[col], kde=True, color='green')
        plt.title(col)

    plt.tight_layout()
    plt.show()




import numpy as np
import pandas as pd

def cap_outliers_for_columns(df, columns=["tenure", "MonthlyCharges", "TotalCharges"], factor=1.5, verbose=True):
    """
    Detects and caps outliers (winsorization) for a list of numeric columns
    using the IQR method.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list
        List of numeric columns to detect & cap outliers.
    factor : float
        IQR multiplier (1.5 = standard, 3.0 = conservative).
    verbose : bool
        Print detailed results.

    Returns
    -------
    df_clean : pandas.DataFrame
        DataFrame with capped outliers.
    summary_df : pandas.DataFrame
        Summary table for outlier handling.
    """
    
    df_clean = df.copy()
    summary_records = []

    for col in columns:
        if col not in df_clean.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        if df_clean[col].dtype not in ["float64", "int64"]:
            raise ValueError(f"Column '{col}' must be numeric.")

        # Compute IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        # Count outliers
        lower_outliers = (df_clean[col] < lower).sum()
        upper_outliers = (df_clean[col] > upper).sum()
        total_outliers = lower_outliers + upper_outliers

        # Cap values
        df_clean[col] = df_clean[col].clip(lower, upper)

        # Add to summary
        summary_records.append({
            "column": col,
            "lower_bound": lower,
            "upper_bound": upper,
            "lower_outliers": lower_outliers,
            "upper_outliers": upper_outliers,
            "total_outliers": total_outliers
        })


    return df_clean





def split_and_scale(df, target_col='Churn'):
    """
    Separates features and target column, then applies StandardScaler
    to all feature columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input processed DataFrame.
    target_col : str
        Name of the target column.

    Returns
    -------
    X_scaled : pandas.DataFrame
        Scaled feature matrix.
    y : pandas.Series
        Target variable.
    scaler : StandardScaler
        Fitted scaler for future use on new data.
    """

    # Separate X and y
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)

    # Convert back to DataFrame with feature names
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)
    return X_scaled, y, scaler

import os

def save_dataframe(df, output_path):
    """
    Saves a DataFrame to a CSV file and creates directories if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save.
    output_path : str
        Path (relative or absolute) to save CSV file.

    Returns
    -------
    None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"✅ Data saved to: {output_path}")
    
    
    

# Preprocessing Pipeline documentation and return df clean  

import pandas as pd

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing step used by the API:
      - convert TotalCharges to numeric and fill missing
      - drop customerID
      - encode categoricals
      - cap outliers
    Returns a cleaned, fully numeric DataFrame (no saving to disk).
    """

    # 1) Fix TotalCharges
    df = convert_to_numeric(df, "TotalCharges", fill_method='median')

    # 2) Drop ID column if exists
    df = drop_column(df, "customerID")

    # 3) Encode categorical features
    # Make sure your encode_categorical_features returns (df_encoded, label_encoders)
    df_encoded, _ = encode_categorical_features(df)

    # 4) Cap outliers for numeric columns
    cols_to_cap = ["tenure", "MonthlyCharges", "TotalCharges"]  # or whatever you used
    df_clean, _ = cap_outliers_for_columns(df_encoded, columns=cols_to_cap, factor=1.5, verbose=False)

    return df_clean

    
    