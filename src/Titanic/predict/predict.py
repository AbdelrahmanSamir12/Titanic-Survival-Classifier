import mlflow
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# === CONFIGURATION ===
MLFLOW_TRACKING_URI = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/mlops.mlflow"  # change 'mlops' if needed
MODEL_NAME = "titanic"
MODEL_ALIAS = "1"  # Only load model marked as "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load Production model from MLflow DagsHub Model Registry
model = mlflow.sklearn.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_ALIAS}")


example_X = np.array([[3, 0, 22.0, 1, 0, 7.25, 0]]) 

y_pred = model.predict(example_X)
# Map numeric prediction to text
label_map = {0: "Not Survived", 1: "Survived"}
text_pred = label_map.get(int(y_pred[0]), "Unknown")

print(f"Prediction: {y_pred[0]} ({text_pred})")