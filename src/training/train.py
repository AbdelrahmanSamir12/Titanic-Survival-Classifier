from functools import partial
import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import cross_validate

from src.Titanic.estimator.random_forest import TitanicRandomForestEstimator
from src.Titanic.estimator.logistic_regression import TitanicLogisticRegressionEstimator

MODEL_PATH = "models"
N_FOLDS = 5
MAX_EVALS = 10

# Define search spaces for each model
MODEL_SPACES = {
    "random_forest": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 300, 50)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "random_state": 42
    },
    "logistic_regression": {
        "C": hp.loguniform("C", -5, 2),
        "penalty": hp.choice("penalty", ["l2"]),
        "solver": hp.choice("solver", ["lbfgs", "liblinear"]),
        "random_state": 42
    }
}

MODEL_CLASSES = {
    "random_forest": TitanicRandomForestEstimator,
    "logistic_regression": TitanicLogisticRegressionEstimator
}


def encode_target(y: pd.Series, model_name: str) -> Tuple[pd.Series, Dict]:
    encoder = {class_: idx for idx, class_ in enumerate(y.unique())}
    decoder = {idx: class_ for class_, idx in encoder.items()}
    label_translator = {"encoder": encoder, "decoder": decoder}
    
    os.makedirs(os.path.join(MODEL_PATH, model_name), exist_ok=True)
    with open(os.path.join(MODEL_PATH, model_name, "target_translator.pkl"), "wb") as f:
        pickle.dump(label_translator, f)
    
    return y.map(encoder), label_translator


def objective(params: Dict[str, Any], X, y, model_class) -> Dict[str, Any]:
    model = model_class(**params)
    scores = cross_validate(model, X, y, cv=N_FOLDS, scoring="accuracy", n_jobs=-1)
    score = np.mean(scores["test_score"])
    return {"loss": -score, "params": params, "status": STATUS_OK}


def train_model(X: pd.DataFrame, y: pd.Series, model_name: str, logger) -> None:
    logger.info(f"Starting training for {model_name}")
    
    # Encode target
    y_encoded, translator = encode_target(y, model_name)
    
    # Setup optimization
    space = MODEL_SPACES[model_name]
    model_class = MODEL_CLASSES[model_name]
    fmin_objective = partial(objective, X=X, y=y_encoded, model_class=model_class)
    trials = Trials()
    
    logger.info("Running hyperparameter optimization")
    best = fmin(
        fn=fmin_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials,
        rstate=np.random.RandomState(42)
    )
    
    # Get best parameters
    best_params = trials.best_trial["result"]["params"]
    logger.info(f"Best parameters: {best_params}")
    
    # Train final model
    final_model = model_class(**best_params)
    final_model.fit(X, y_encoded)
    
    # Save model
    os.makedirs(os.path.join(MODEL_PATH, model_name), exist_ok=True)
    with open(os.path.join(MODEL_PATH, model_name, "model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
    
    logger.info(f"Training completed for {model_name}")