from src.Titanic.data.process import (
    load_raw_data,
    preprocess_data,
    split_data,
    save_processed_data,
    load_processed_data
)
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.Titanic.logger import TitanicLogger


def main():
    # Setup logger
    logger = TitanicLogger(logs_path="titanic", level="DEBUG")
    
    # Data processing
    logger.info("Loading and preprocessing data")
    df = load_raw_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_processed_data(X_train, X_test, y_train, y_test)
    
    # Model training
    for model_name in ["random_forest", "logistic_regression"]:
        train_model(X_train, y_train, model_name, logger)
        evaluate_model(X_test, y_test, model_name, logger)


if __name__ == "__main__":
    main()