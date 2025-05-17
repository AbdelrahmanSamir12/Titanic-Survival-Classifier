import os
import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from src.Titanic.training.preprocessing import Preprocessor
from src.Titanic.training.train import train_model
from src.Titanic.training.evaluate import evaluate_model
from src.Titanic.logger import TitanicLogger
import joblib
import mlflow
import dagshub
from sklearn.pipeline import Pipeline


from dotenv import load_dotenv
load_dotenv()

@hydra.main(version_base=None, config_path="configs", config_name="main")
def run_pipeline(cfg: DictConfig):
    logger = TitanicLogger(logs_path="logs", level="DEBUG")

    # Start MLflow run
    # === DAGsHub & MLflow Setup ===
    model = hydra.utils.instantiate(cfg.model)
    
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=cfg.tracking.repo_name,
        mlflow=cfg.tracking.use_mlflow
    )

    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    mlflow.sklearn.autolog(log_models=True)
    #client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    with mlflow.start_run(run_name="Titanic_Classifier_Run"):
        # Log config params
        mlflow.log_params({
            "test_size": cfg.data.test_size,
            "random_state": cfg.data.random_state,
            "model_type": cfg.model._target_,
            # Add more config params if needed
        })
        
        # Load raw data
        logger.info("Loading raw data")
        df = pd.read_csv(cfg.data.raw_path)
        
        # Split features and target
        X = df[cfg.data.features]
        y = df[cfg.data.target]
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_state,
            stratify=y
        )
        
        # Preprocessing
        logger.info("Running preprocessing")
        preprocessor = Preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save preprocessing pipeline
        preprocessor_path = os.path.join(cfg.data.processed_path, "preprocessor.joblib")
        preprocessor.save(preprocessor_path)
        
        # Log preprocessor as artifact
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        
        # Train model
        logger.info("Training model")
        model = train_model(X_train_processed, y_train, cfg)
        
        # Save model
        model_path = os.path.join(cfg.data.processed_path, "model.joblib")
        joblib.dump(model, model_path)
        
        # Log model to MLflow
        #mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Evaluate model
        #logger.info("Evaluating model")
        accuracy = evaluate_model(X_test_processed, y_test, cfg.model._target_.split(".")[-1], logger)
        
        # Combine preprocessor + model
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor.full_pipeline),  # your sklearn pipeline inside Preprocessor
            ('model', model)
        ])
        # Log the full pipeline as MLflow model
        mlflow.sklearn.log_model(
            full_pipeline,
            artifact_path="full_pipeline_model",
            signature=mlflow.models.infer_signature(X_train, y_train)  # ✅ Explicitly match raw input
        )
        joblib.dump(full_pipeline, "full_pipeline.joblib")
        # Log metrics
        #mlflow.log_metric("accuracy", accuracy)
        
        logger.info(f"Pipeline and model logged to MLflow")

if __name__ == "__main__":
    run_pipeline()

    


