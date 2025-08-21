import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# File paths
RAW_CSV = "data/raw/data.csv"
PROCESSED_CSV = "data/processed/transactions_processed.csv"

def ensure_paths():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV}")
    os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)

def load_data(raw_csv_path):
    print(f"Loading raw data from {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    print(f"Loaded raw data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def compute_rfm(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    rfm_df = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()

    df = df.merge(rfm_df, on='CustomerId', how='left')
    return df

def encode_categoricals(df):
    cat_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 
                'ProductCategory', 'ChannelId', 'PricingStrategy', 'SubscriptionId']
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = pd.DataFrame(
        ohe.fit_transform(df[cat_cols]), 
        columns=ohe.get_feature_names_out(cat_cols)
    )
    
    df = df.drop(columns=cat_cols).reset_index(drop=True)
    df = pd.concat([df, encoded], axis=1)
    return df

def process_data(raw_csv_path=RAW_CSV, processed_csv_path=PROCESSED_CSV):
    ensure_paths()
    df = load_data(raw_csv_path)
    df = compute_rfm(df)
    df = encode_categoricals(df)
    df.to_csv(processed_csv_path, index=False)
    print(f"Processed data saved to {processed_csv_path}")

if __name__ == "__main__":
    process_data()
