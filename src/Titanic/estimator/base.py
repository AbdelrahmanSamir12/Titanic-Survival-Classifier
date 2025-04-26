import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y


class BaseTitanicEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        raise NotImplementedError("predict_proba must be implemented in subclass")

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)