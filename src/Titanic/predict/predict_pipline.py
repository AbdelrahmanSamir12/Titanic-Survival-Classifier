import mlflow
import pandas as pd
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
load_dotenv()

# === CONFIGURATION ===
MLFLOW_TRACKING_URI = f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/mlops.mlflow"
MODEL_NAME = "titanic_full_pipline"
MODEL_ALIAS = "2"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Load the **FULL pipeline** (Preprocessor + Model)
#loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_ALIAS}")

# Load the model from MLflow
logged_model_uri = "runs:/9d5004d8f9754ce4a8bb4dd5168303dd/full_pipeline_model"
model = mlflow.pyfunc.load_model(logged_model_uri)

print("==============")
model_uri = f"models:/{MODEL_NAME}/{MODEL_ALIAS}"
model_info = mlflow.model.get_model_info(model_uri)
print(model_info.signature)
print("==============")
# === Prepare raw Titanic input ===
# Now it's safe to pass raw features (because pipeline will process them)
# Example input (dict form)
input_data = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 34.5,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.8292,
    "Embarked": "Q"
}])



# ✅ Predict using full pipeline (raw input is ok)
y_pred = model.predict(input_data)

label_map = {0: "Not Survived", 1: "Survived"}
text_pred = label_map.get(int(y_pred[0]), "Unknown")

print(f"Prediction: {y_pred[0]} ({text_pred})")