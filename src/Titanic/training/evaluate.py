import json
import os
import joblib
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix)

def evaluate_model(X_test, y_test, model_name: str, logger):
    """Evaluate model and save metrics"""
    logger.info(f"Evaluating {model_name} model")
    
    # Load model
    model = joblib.load(f"models/{model_name}.joblib")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Save metrics
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"{model_name} evaluation metrics:")
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            logger.info(f"{metric}: {value:.4f}")
    
    logger.info("Confusion Matrix:")
    logger.info("\n" + str(confusion_matrix(y_test, y_pred)))