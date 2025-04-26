import json
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

REPORT_PATH = "reports"


def evaluate_model(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    logger
) -> Dict[str, float]:
    logger.info(f"Evaluating {model_name}")
    
    # Load model and translator
    with open(os.path.join("models", model_name, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join("models", model_name, "target_translator.pkl"), "rb") as f:
        translator = pickle.load(f)
    
    # Encode target if needed
    y_test_encoded = y_test.map(translator["encoder"])
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test_encoded, y_pred),
        "precision": precision_score(y_test_encoded, y_pred),
        "recall": recall_score(y_test_encoded, y_pred),
        "f1": f1_score(y_test_encoded, y_pred),
        "roc_auc": roc_auc_score(y_test_encoded, y_proba),
        "confusion_matrix": confusion_matrix(y_test_encoded, y_pred).tolist()
    }
    
    # Save report
    os.makedirs(os.path.join(REPORT_PATH, model_name), exist_ok=True)
    with open(os.path.join(REPORT_PATH, model_name, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation completed for {model_name}")
    return metrics