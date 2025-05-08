import sys
import pandas as pd
import joblib

# Load the saved pipeline
pipeline = joblib.load("full_pipeline.joblib")

# Example: Load input from a CSV file, or define manually
# If run as: python predict.py input.csv
if len(sys.argv) > 1:
    input_file = sys.argv[1]
    X_new = pd.read_csv(input_file)
else:
    # Manual sample (make sure columns match your training columns!)
    X_new = pd.DataFrame([{
        # Example for Titanic: fill with your actual feature names/values!
        "Pclass": 3,
        "Sex": "male",
        "Age": 29,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }])

# Predict
preds = pipeline.predict(X_new)
print("Predictions:", preds)

# Optional: output probabilities
# probs = pipeline.predict_proba(X_new)
# print("Probabilities:", probs)