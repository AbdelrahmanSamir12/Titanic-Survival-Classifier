import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(X_train, y_train, model_name: str, logger):
    """Train specified model and save it"""
    logger.info(f"Training {model_name} model")
    
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    model = models[model_name]
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.joblib")
    logger.info(f"{model_name} model trained and saved")