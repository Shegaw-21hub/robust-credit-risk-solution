import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from datetime import datetime
import os
import warnings
from pandas.api.types import is_datetime64tz_dtype as is_tz_aware

# Suppress all warnings
warnings.filterwarnings("ignore")

class RFMTransformer(BaseEstimator, TransformerMixin):
    """Calculate RFM features and temporal features"""
    def __init__(self, snapshot_date=None):
        self.snapshot_date = pd.to_datetime(snapshot_date) if snapshot_date else datetime.now()
        # Check for timezone awareness using non-deprecated method
        if hasattr(self.snapshot_date, 'tz') and self.snapshot_date.tz is not None:
            self.snapshot_date = self.snapshot_date.tz_localize(None)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert to datetime and ensure timezone-naive
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        # Check for timezone awareness using non-deprecated method
        if hasattr(X['TransactionStartTime'].dtype, 'tz'):
            X['TransactionStartTime'] = X['TransactionStartTime'].dt.tz_localize(None)
        
        # Calculate RFM metrics
        rfm = X.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': ['sum', 'mean', 'std']
        })
        rfm.columns = ['Recency', 'Frequency', 'MonetarySum', 'MonetaryMean', 'MonetaryStd']
        rfm['MonetaryStd'] = rfm['MonetaryStd'].fillna(0)
        
        # Extract temporal features
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        
        return pd.merge(X, rfm, on='CustomerId')

class HighRiskLabelGenerator(BaseEstimator, TransformerMixin):
    """Create binary risk labels using KMeans clustering"""
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.high_risk_cluster_ = None
        
    def fit(self, X, y=None):
        rfm_features = X[['Recency', 'Frequency', 'MonetarySum']]
        X_scaled = self.scaler.fit_transform(rfm_features)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X_scaled)
        
        # Identify highest risk cluster
        cluster_stats = X.assign(Cluster=self.kmeans.labels_)
        risk_order = cluster_stats.groupby('Cluster')[['Recency', 'Frequency', 'MonetarySum']].mean()
        self.high_risk_cluster_ = risk_order.sort_values(
            ['Recency', 'Frequency', 'MonetarySum'],
            ascending=[False, True, True]
        ).index[0]
        return self
    
    def transform(self, X):
        rfm_features = X[['Recency', 'Frequency', 'MonetarySum']]
        X_scaled = self.scaler.transform(rfm_features)
        clusters = self.kmeans.predict(X_scaled)
        X['is_high_risk'] = (clusters == self.high_risk_cluster_).astype(int)
        return X

def build_feature_pipeline():
    """Build complete feature engineering pipeline"""
    # Numerical features pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Define feature columns
    numerical_features = ['Recency', 'Frequency', 'MonetarySum', 'MonetaryMean', 'MonetaryStd']
    categorical_features = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    temporal_features = ['TransactionHour', 'TransactionDay', 'TransactionMonth']
    
    # Column transformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features),
        ('temp', 'passthrough', temporal_features)
    ])
    
    # Complete pipeline
    return Pipeline([
        ('rfm_features', RFMTransformer()),
        ('risk_labels', HighRiskLabelGenerator()),
        ('preprocessing', preprocessor)
    ])

def main():
    """Main execution function"""
    try:
        # Get the absolute path to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set up normalized paths
        processed_dir = os.path.normpath(os.path.join(script_dir, '../data/processed'))
        os.makedirs(processed_dir, exist_ok=True)
        
        input_path = os.path.normpath(os.path.join(script_dir, '../data/raw/data.csv'))
        output_path = os.path.normpath(os.path.join(processed_dir, 'model_ready_data.csv'))
        
        # Load data
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found at {input_path}")
        data = pd.read_csv(input_path)
        
        # Process data
        pipeline = build_feature_pipeline()
        processed_data = pipeline.fit_transform(data)
        
        # Save results
        pd.DataFrame(processed_data).to_csv(output_path, index=False)
        print(f"Data successfully saved to:\n{output_path}")
        print(f"File created successfully ({os.path.getsize(output_path):,} bytes)")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Disable parallel processing warnings
    os.environ['LOKY_MAX_CPU_COUNT'] = '1'
    main()