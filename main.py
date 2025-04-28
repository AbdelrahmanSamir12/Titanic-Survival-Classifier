import hydra
from omegaconf import DictConfig
from src.Titanic.data.process import (
    load_raw_data,
    preprocess_data,
    split_data,
    save_processed_data
)
from src.Titanic.training.train import train_model
from src.Titanic.training.evaluate import evaluate_model
from src.Titanic.logger import TitanicLogger

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    # Setup logger
    logger = TitanicLogger(logs_path="logs", level="DEBUG")
    
    # Data processing
    logger.info("Loading and preprocessing data")
    df = load_raw_data(cfg)
    df = preprocess_data(df, cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    save_processed_data(X_train, X_test, y_train, y_test, cfg)
    
    # Model training and evaluation
    logger.info("Training model")
    model = train_model(X_train, y_train, cfg)
    
    logger.info("Evaluating model")
    evaluate_model(X_test, y_test, cfg.model._target_.split(".")[-1])

if __name__ == "__main__":
    main()