import os
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

import mlflow
import mlflow.sklearn
import dagshub

from src.Titanic.logger import TitanicLogger

from dotenv import load_dotenv
load_dotenv()

@hydra.main(version_base=None, config_path="../../../configs", config_name="main")
def main(cfg: DictConfig):
    from src.Titanic.data.process import load_raw_data, preprocess_data, split_data

    logger = TitanicLogger(logs_path="logs", level="DEBUG")

    # === DAGsHub & MLflow Setup ===
    model = hydra.utils.instantiate(cfg.model)
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=cfg.tracking.repo_name,
        mlflow=cfg.tracking.use_mlflow
    )
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    #client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # Track experiment in MLflow
    mlflow.set_experiment("Titanic-GridSearch")
    mlflow.sklearn.autolog()

    # --- Data pipeline ---
    df = load_raw_data(cfg)
    df = preprocess_data(df, cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    # --- Grid Search ---
    model = hydra.utils.instantiate(cfg.model)
    param_grid = OmegaConf.to_container(cfg.grid_search.param_grid, resolve=True)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Log test performance manually (autolog only logs CV!)
    y_pred = grid_search.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    #mlflow.log_metric("test_accuracy", test_acc)

    # Log best estimator as artifact
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/grid_search_best.joblib"
    joblib.dump(grid_search.best_estimator_, best_model_path)
    #mlflow.log_artifact(best_model_path)

    logger.info(f"Best parameters:{grid_search.best_params_}")
    logger.info(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
    main()