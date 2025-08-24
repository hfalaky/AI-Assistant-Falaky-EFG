# preprocessing.py
import pandas as pd
import numpy as np

def preprocess_portfolio(file_path="Active_Clients_Portfolio.csv"):
    # by default it takes active clients portfolio as file path, but if given an argument it will overwrite
    """
    Preprocess the merged client portfolio dataset:
      - Load CSV
      - Standardize column names
      - Handle missing values
      - Drop unusable rows
      - Enforce correct data types
      - Return clean DataFrame
    """

    # 1. Load dataset
    df = pd.read_csv(file_path)
    print(f"Initial shape: {df.shape}")

    # 2. Standardize column names (lowercase, underscores)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # 3. Drop rows missing critical IDs
    df = df.dropna(subset=["clientid", "clientaccprofileid"])

        # Also keep only active traders (HasTrades2024 must be True/1)
    if "hastrades2024" in df.columns:
        df = df[df["hastrades2024"] == True]  # or == 1 depending on encoding

    print(f"After dropping missing IDs & filtering HasTrades2024: {df.shape}")

    # 4. Handle missing numeric values → impute with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # 5. Handle missing categorical values → fill with "Unknown"
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # 6. Optional: remove duplicate clients (keep first)
    df = df.drop_duplicates(subset=["clientid", "clientaccprofileid"])

    # 7. Enforce correct data types for this portfolio format
    dtype_map = {
        # IDs
        "clientid": str,
        "clientaccprofileid": str,

        # Dates
        "clientsincedate_x": "datetime64[ns]",
        "clientsincedate_y": "datetime64[ns]",
        "checkpoint1": "datetime64[ns]",
        "checkpoint2": "datetime64[ns]",
        "interval_start": "datetime64[ns]",
        "interval_end": "datetime64[ns]",

        # Booleans
        "hastrades2024": bool,

        # Numerics
        "daysasclient": "Int64",
        "age": "Int64",
        "sourceid": "Int64",
        "numprofiles": "Int64",
        "netroi": float,
        "mostprofitablesecurityroi": float,
        "mostprofitablesectorroi": float,
        "tradesinthemostactivemonth": "Int64",
        "totaltradesin24": "Int64",
        "totaltradesvolumein24": float,
        "numberoftradesonmosttradedsecurity": "Int64",
        "tradesvolumeofmosttradedsecurity": float,
        "tradesvolumeofmosttradedsector": float,
        "numberoftradesinmosttradedsector": "Int64",
        "durationheld": "Int64",
    }

    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype, errors="ignore")
            except Exception as e:
                print(f"⚠️ Could not convert {col} to {dtype}: {e}")

    print(f"Final shape after preprocessing: {df.shape}")
    return df


if __name__ == "__main__": #wont run this part if imported cuz name= main when runs directly
    cleaned_df = preprocess_portfolio()
    cleaned_df.to_csv("Cleaned_Active_Portfolio.csv", index=False)
    print("✅ Cleaned portfolio saved as Cleaned_Active_Portfolio.csv")
