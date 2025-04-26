import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(file_name: str = "train") -> pd.DataFrame:
    return pd.read_csv(os.path.join("data", "raw", f"{file_name}.csv"))


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Feature engineering
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Drop columns
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    return df


def split_data(df: pd.DataFrame, target: str = "Survived") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def save_processed_data(X_train, X_test, y_train, y_test):
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_parquet("data/processed/X_train.parquet")
    X_test.to_parquet("data/processed/X_test.parquet")
    y_train.to_parquet("data/processed/y_train.parquet")
    y_test.to_parquet("data/processed/y_test.parquet")


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_parquet("data/processed/X_train.parquet")
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet")
    y_test = pd.read_parquet("data/processed/y_test.parquet")
    return X_train, X_test, y_train, y_test