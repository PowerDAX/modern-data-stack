"""
Shared Data Connector Module

This module provides common data connection patterns and utilities for notebooks.
Includes database connections, file loading, and API integrations.
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime
import yaml
import json

# Database connectors
try:
    import psycopg2
    from sqlalchemy import create_engine
    import sqlite3
    HAS_DB_SUPPORT = True
except ImportError:
    HAS_DB_SUPPORT = False
    warnings.warn("Database support not available. Install psycopg2 and sqlalchemy.")

# Cloud storage connectors
try:
    import boto3
    from azure.storage.blob import BlobServiceClient
    from google.cloud import storage
    HAS_CLOUD_SUPPORT = True
except ImportError:
    HAS_CLOUD_SUPPORT = False
    warnings.warn("Cloud storage support not available. Install boto3, azure-storage-blob, google-cloud-storage.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataConnector:
    """Base class for data connectors"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.connection = None
        
    def connect(self):
        """Establish connection"""
        raise NotImplementedError("Subclasses must implement connect method")
        
    def disconnect(self):
        """Close connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class DatabaseConnector(DataConnector):
    """Database connector for SQL databases"""
    
    def __init__(self, connection_string: str, **kwargs):
        super().__init__()
        self.connection_string = connection_string
        self.engine = None
        
    def connect(self):
        """Connect to database"""
        if not HAS_DB_SUPPORT:
            raise ImportError("Database support not available")
            
        try:
            self.engine = create_engine(self.connection_string)
            self.connection = self.engine.connect()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def query(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        try:
            df = pd.read_sql_query(sql, self.engine, params=params)
            logger.info(f"Query executed successfully. Rows: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute(self, sql: str, params: Optional[Dict] = None) -> None:
        """Execute SQL statement"""
        try:
            self.connection.execute(sql, params or {})
            logger.info("Statement executed successfully")
        except Exception as e:
            logger.error(f"Statement execution failed: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get table schema information"""
        sql = f"""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        return self.query(sql)
    
    def list_tables(self) -> List[str]:
        """List all tables in the database"""
        sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        df = self.query(sql)
        return df['table_name'].tolist()


class FileConnector(DataConnector):
    """File connector for various file formats"""
    
    def __init__(self, base_path: str = "."):
        super().__init__()
        self.base_path = Path(base_path)
        
    def connect(self):
        """Validate base path"""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")
        logger.info(f"File connector initialized. Base path: {self.base_path}")
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        full_path = self.base_path / file_path
        try:
            df = pd.read_csv(full_path, **kwargs)
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"CSV loading failed: {e}")
            raise
    
    def load_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load Excel file"""
        full_path = self.base_path / file_path
        try:
            df = pd.read_excel(full_path, **kwargs)
            logger.info(f"Excel loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Excel loading failed: {e}")
            raise
    
    def load_json(self, file_path: str, **kwargs) -> Union[pd.DataFrame, Dict]:
        """Load JSON file"""
        full_path = self.base_path / file_path
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
                logger.info(f"JSON loaded as DataFrame. Shape: {df.shape}")
                return df
            else:
                logger.info(f"JSON loaded as dictionary. Keys: {list(data.keys())}")
                return data
        except Exception as e:
            logger.error(f"JSON loading failed: {e}")
            raise
    
    def load_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file"""
        full_path = self.base_path / file_path
        try:
            df = pd.read_parquet(full_path, **kwargs)
            logger.info(f"Parquet loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Parquet loading failed: {e}")
            raise
    
    def save_csv(self, df: pd.DataFrame, file_path: str, **kwargs) -> None:
        """Save DataFrame to CSV"""
        full_path = self.base_path / file_path
        try:
            df.to_csv(full_path, index=False, **kwargs)
            logger.info(f"CSV saved successfully. Path: {full_path}")
        except Exception as e:
            logger.error(f"CSV saving failed: {e}")
            raise
    
    def save_parquet(self, df: pd.DataFrame, file_path: str, **kwargs) -> None:
        """Save DataFrame to Parquet"""
        full_path = self.base_path / file_path
        try:
            df.to_parquet(full_path, index=False, **kwargs)
            logger.info(f"Parquet saved successfully. Path: {full_path}")
        except Exception as e:
            logger.error(f"Parquet saving failed: {e}")
            raise


class CloudStorageConnector(DataConnector):
    """Cloud storage connector for AWS S3, Azure Blob, Google Cloud Storage"""
    
    def __init__(self, provider: str, **kwargs):
        super().__init__()
        self.provider = provider.lower()
        self.credentials = kwargs
        self.client = None
        
    def connect(self):
        """Connect to cloud storage"""
        if not HAS_CLOUD_SUPPORT:
            raise ImportError("Cloud storage support not available")
            
        try:
            if self.provider == 'aws':
                self.client = boto3.client('s3', **self.credentials)
            elif self.provider == 'azure':
                self.client = BlobServiceClient(**self.credentials)
            elif self.provider == 'gcp':
                self.client = storage.Client(**self.credentials)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
            logger.info(f"Connected to {self.provider} cloud storage")
        except Exception as e:
            logger.error(f"Cloud storage connection failed: {e}")
            raise
    
    def list_files(self, bucket: str, prefix: str = "") -> List[str]:
        """List files in cloud storage"""
        try:
            if self.provider == 'aws':
                response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
                return [obj['Key'] for obj in response.get('Contents', [])]
            elif self.provider == 'azure':
                container_client = self.client.get_container_client(bucket)
                return [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]
            elif self.provider == 'gcp':
                bucket_obj = self.client.bucket(bucket)
                return [blob.name for blob in bucket_obj.list_blobs(prefix=prefix)]
        except Exception as e:
            logger.error(f"File listing failed: {e}")
            raise
    
    def download_file(self, bucket: str, key: str, local_path: str) -> None:
        """Download file from cloud storage"""
        try:
            if self.provider == 'aws':
                self.client.download_file(bucket, key, local_path)
            elif self.provider == 'azure':
                blob_client = self.client.get_blob_client(container=bucket, blob=key)
                with open(local_path, 'wb') as f:
                    f.write(blob_client.download_blob().readall())
            elif self.provider == 'gcp':
                bucket_obj = self.client.bucket(bucket)
                blob = bucket_obj.blob(key)
                blob.download_to_filename(local_path)
                
            logger.info(f"File downloaded successfully: {local_path}")
        except Exception as e:
            logger.error(f"File download failed: {e}")
            raise


class ConfigManager:
    """Configuration manager for data connectors"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                logger.warning(f"Configuration file not found: {self.config_file}")
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise
    
    def get_connection_string(self, name: str) -> str:
        """Get database connection string"""
        db_config = self.config.get('databases', {}).get(name, {})
        if not db_config:
            raise ValueError(f"Database configuration not found: {name}")
        
        return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    def get_cloud_credentials(self, provider: str) -> Dict:
        """Get cloud storage credentials"""
        return self.config.get('cloud_storage', {}).get(provider, {})


# Utility functions
def get_sample_data(dataset_name: str = 'iris') -> pd.DataFrame:
    """Get sample datasets for testing"""
    from sklearn.datasets import load_iris, load_boston, load_wine
    
    if dataset_name == 'iris':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == 'boston':
        data = load_boston()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == 'wine':
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_mock_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
    """Create mock data for testing"""
    np.random.seed(42)
    
    # Generate features
    data = {}
    for i in range(n_features):
        if i % 3 == 0:  # Categorical
            data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
        elif i % 3 == 1:  # Numerical
            data[f'num_feature_{i}'] = np.random.normal(0, 1, n_samples)
        else:  # Integer
            data[f'int_feature_{i}'] = np.random.randint(0, 100, n_samples)
    
    # Add target variable
    data['target'] = np.random.randint(0, 2, n_samples)
    
    df = pd.DataFrame(data)
    logger.info(f"Mock data created. Shape: {df.shape}")
    return df


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """Validate DataFrame structure and quality"""
    validation_results = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'validation_passed': True,
        'issues': []
    }
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
            validation_results['validation_passed'] = False
    
    # Check for excessive missing values
    high_missing = df.isnull().sum() / len(df) > 0.5
    if high_missing.any():
        high_missing_cols = high_missing[high_missing].index.tolist()
        validation_results['issues'].append(f"High missing values (>50%): {high_missing_cols}")
    
    # Check for duplicate rows
    if validation_results['duplicate_rows'] > 0:
        validation_results['issues'].append(f"Found {validation_results['duplicate_rows']} duplicate rows")
    
    return validation_results


# Example usage and testing
if __name__ == "__main__":
    # Test file connector
    print("Testing File Connector:")
    with FileConnector() as fc:
        mock_df = create_mock_data(100, 5)
        fc.save_csv(mock_df, "test_data.csv")
        loaded_df = fc.load_csv("test_data.csv")
        print(f"Loaded data shape: {loaded_df.shape}")
    
    # Test sample data
    print("\nTesting Sample Data:")
    iris_df = get_sample_data('iris')
    print(f"Iris dataset shape: {iris_df.shape}")
    
    # Test validation
    print("\nTesting Validation:")
    validation_results = validate_dataframe(iris_df, required_columns=['target'])
    print(f"Validation passed: {validation_results['validation_passed']}")
    print(f"Issues: {validation_results['issues']}") 