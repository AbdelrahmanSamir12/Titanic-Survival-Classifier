import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

DATA_PATH = "/teamspace/studios/this_studio/mine/mlops/data"

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects specified features"""
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.features]

class Preprocessor:
    def __init__(self):
        self.numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
        self.categorical_features = ['Pclass', 'Sex', 'Embarked']
        self.target = 'Survived'
        self.features = self.numeric_features + self.categorical_features
        
        # Define preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)])
        
        self.full_pipeline = Pipeline(steps=[
            ('feature_selector', FeatureSelector(self.features)),
            ('preprocessor', self.preprocessor)])
    
    def fit_transform(self, X, y=None):
        return self.full_pipeline.fit_transform(X, y)
    
    def transform(self, X):
        return self.full_pipeline.transform(X)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.full_pipeline, path)
    
    @classmethod
    def load(cls, path):
        preprocessor = cls()
        preprocessor.full_pipeline = joblib.load(path)
        return preprocessor

def run_preprocessing():
    """Run complete preprocessing and save pipeline"""
    raw_path = os.path.join(DATA_PATH, "raw/train.csv")
    processed_path = os.path.join(DATA_PATH, "processed")
    pipeline_path = os.path.join(DATA_PATH, "processed/preprocessor.joblib")
    
    # Load and preprocess data
    df = pd.read_csv(raw_path)
    preprocessor = Preprocessor()
    X = df.drop(preprocessor.target, axis=1)
    y = df[preprocessor.target]
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X, y)
    
    # Save processed data and pipeline
    os.makedirs(processed_path, exist_ok=True)
    #pd.DataFrame(X_processed).to_parquet(os.path.join(processed_path, "X_train.parquet"))
    #y.to_parquet(os.path.join(processed_path, "y_train.parquet"))
    preprocessor.save(pipeline_path)
    
    print(f"Preprocessing complete. Pipeline saved to {pipeline_path}")


if __name__ == "__main__":
    run_preprocessing()