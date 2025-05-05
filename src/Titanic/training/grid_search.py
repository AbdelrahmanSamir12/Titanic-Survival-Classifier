import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import os


@hydra.main(version_base=None, config_path="../../../configs", config_name="main")
def main(cfg: DictConfig):
    from src.Titanic.data.process import load_raw_data, preprocess_data, split_data

    # Load and process data
    df = load_raw_data(cfg)
    df = preprocess_data(df, cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    # Model and param_grid
    model = hydra.utils.instantiate(cfg.model)
    param_grid = cfg.grid_search.param_grid
    param_grid = OmegaConf.to_container(cfg.grid_search.param_grid, resolve=True)
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Save best estimator
    os.makedirs("models", exist_ok=True)
    joblib.dump(grid_search.best_estimator_, "models/grid_search_best.joblib")

    # Evaluate on test
    y_pred = grid_search.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Best parameters:", grid_search.best_params_)
    print("Test accuracy:", acc)

if __name__ == "__main__":
    main()