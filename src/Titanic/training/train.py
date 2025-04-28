import os
import joblib
import hydra
from omegaconf import DictConfig
from sklearn.base import BaseEstimator

def train_model(X_train, y_train, cfg: DictConfig) -> BaseEstimator:
    """Train model using Hydra config"""
    # Initialize model from config
    model = hydra.utils.instantiate(cfg.model)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_name = cfg.model._target_.split(".")[-1]
    joblib.dump(model, f"models/{model_name}.joblib")
    
    return model