import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw_data(file_name: str = "train") -> pd.DataFrame:
    """Load raw data from CSV"""
    return pd.read_csv(os.path.join("data", "raw", f"{file_name}.csv"))

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform essential preprocessing"""
    # Feature selection and engineering
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    # Convert categorical features
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, None: 0})
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(0, inplace=True)
    
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Split data into train and test sets"""
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def save_processed_data(X_train, X_test, y_train, y_test):
    """Save processed data to parquet"""
    os.makedirs("data/processed", exist_ok=True)
    
    # Convert Series to DataFrame before saving
    pd.DataFrame(X_train).to_parquet("data/processed/X_train.parquet")
    pd.DataFrame(X_test).to_parquet("data/processed/X_test.parquet")
    pd.DataFrame(y_train).to_parquet("data/processed/y_train.parquet")
    pd.DataFrame(y_test).to_parquet("data/processed/y_test.parquet")

def load_processed_data() -> tuple:
    """Load processed data from parquet"""
    X_train = pd.read_parquet("data/processed/X_train.parquet")
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet")['Survived']
    y_test = pd.read_parquet("data/processed/y_test.parquet")['Survived']
    return X_train, X_test, y_train, y_test