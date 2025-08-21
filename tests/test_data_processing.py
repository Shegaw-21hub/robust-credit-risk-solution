# tests/test_data_processing.py
import pytest
import pandas as pd
import src.data_processing as dp  # Make sure sys.path includes src in your main code

@pytest.fixture
def sample_data():
    data = {
        'CustomerId': [1, 1, 2],
        'TransactionId': [101, 102, 201],
        'TransactionStartTime': ['2025-08-20', '2025-08-19', '2025-08-18'],
        'Amount': [100, 150, 200],
        'CurrencyCode': ['USD', 'USD', 'EUR'],
        'CountryCode': ['US', 'US', 'FR'],
        'ProviderId': ['Prov1', 'Prov1', 'Prov2'],
        'ProductId': ['Prod1', 'Prod2', 'Prod3'],
        'ProductCategory': ['Cat1', 'Cat2', 'Cat3'],
        'ChannelId': ['C1', 'C1', 'C2'],
        'PricingStrategy': ['Fixed', 'Fixed', 'Fixed'],
        'SubscriptionId': ['S1', 'S2', 'S3']
    }
    df = pd.DataFrame(data)
    return df

def test_compute_rfm(sample_data):
    df = dp.compute_rfm(sample_data)
    assert 'Recency' in df.columns
    assert 'Frequency' in df.columns
    assert 'Monetary' in df.columns

    # Check aggregated values per customer
    rfm_df = df.groupby('CustomerId').agg({
        'Recency': 'first',
        'Frequency': 'first',
        'Monetary': 'first'
    })
    assert rfm_df.loc[1, 'Frequency'] == 2
    assert rfm_df.loc[2, 'Frequency'] == 1
    assert rfm_df.loc[1, 'Monetary'] == 250
    assert rfm_df.loc[2, 'Monetary'] == 200

def test_encode_categoricals(sample_data):
    df = dp.encode_categoricals(sample_data)
    # Original categorical columns should be removed
    cat_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 
                'ProductCategory', 'ChannelId', 'PricingStrategy', 'SubscriptionId']
    for col in cat_cols:
        assert col not in df.columns
    # There should be new columns after one-hot encoding
    encoded_cols = [col for col in df.columns if '_' in col]
    assert len(encoded_cols) > 0
