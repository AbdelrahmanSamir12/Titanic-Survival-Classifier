import numpy as np
import litserve as ls
import joblib
import pandas as pd
from src.deployment.online.requests import InferenceRequest

class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        # Load the full sklearn pipeline (preprocessing + model)
        self._model = joblib.load("full_pipeline.joblib")
        # If you want to map 0/1 to class labels, define a decoder:
        self._decoder = {0: "Did not survive", 1: "Survived"}

    def decode_request(self, request):
        try:
            # If "input" not in request, assume request itself is the input
            input_data = request.get("input", request)
            parsed_request = InferenceRequest(**input_data)
            data = {field: getattr(parsed_request, field) for field in parsed_request.__fields__.keys()}
            x = pd.DataFrame([data])
            return x
        except Exception as e:
            print(f"Decode error: {e}")
            return None
    def predict(self, x):
        if x is not None:
            return self._model.predict(x)
        return None

    def encode_response(self, output):
        if output is None:
            message = "Error Occurred"
            prediction = []
        else:
            message = "Response Produced Successfully"
            prediction = [self._decoder[val] for val in output]
        return {
            "message": message,
            "prediction": prediction
        }


# Now you can run with
# uvicorn titanic_api:app --host 0.0.0.0 --port 8000