import os
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

def load_raw_data(cfg: DictConfig) -> pd.DataFrame:
    """Load raw data from CSV using config path"""
    # Handle both nested and flat config structures
    data_cfg = cfg.data.data if hasattr(cfg.data, 'data') else cfg.data
    #print(f"Trying to load from: {data_cfg.raw_path}")
    return pd.read_csv(data_cfg.raw_path)

def preprocess_data(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Perform preprocessing without SettingWithCopyWarning"""
    # Create a copy of the selected columns
    processed_df = df[cfg.data.features + [cfg.data.target]].copy()
    
    # Convert categorical features
    processed_df.loc[:, 'Sex'] = processed_df['Sex'].map({'male': 0, 'female': 1})
    processed_df.loc[:, 'Embarked'] = processed_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, None: 0})
    
    # Handle missing values
    processed_df.loc[:, 'Age'] = processed_df['Age'].fillna(processed_df['Age'].median())
    processed_df.loc[:, 'Fare'] = processed_df['Fare'].fillna(processed_df['Fare'].median())
    processed_df.loc[:, 'Embarked'] = processed_df['Embarked'].fillna(0)
    
    return processed_df

def split_data(df: pd.DataFrame, cfg: DictConfig) -> tuple:
    """Split data using config parameters"""
    X = df.drop(cfg.data.target, axis=1)
    y = df[cfg.data.target]
    return train_test_split(
        X, y, 
        test_size=cfg.data.test_size, 
        random_state=cfg.data.random_state, 
        stratify=y
    )

def save_processed_data(X_train, X_test, y_train, y_test, cfg: DictConfig):
    """Save processed data using config path"""
    os.makedirs(cfg.data.processed_path, exist_ok=True)
    
    # Convert to DataFrame before saving
    pd.DataFrame(X_train).to_parquet(f"{cfg.data.processed_path}/X_train.parquet")
    pd.DataFrame(X_test).to_parquet(f"{cfg.data.processed_path}/X_test.parquet")
    pd.DataFrame(y_train, columns=[cfg.data.target]).to_parquet(f"{cfg.data.processed_path}/y_train.parquet")
    pd.DataFrame(y_test, columns=[cfg.data.target]).to_parquet(f"{cfg.data.processed_path}/y_test.parquet")