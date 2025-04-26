from sklearn.ensemble import RandomForestClassifier
from .base import BaseTitanicEstimator


class TitanicRandomForestEstimator(BaseTitanicEstimator):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, y):
        super().fit(X, y)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X):
        super().predict_proba(X)  # Checks if fitted
        return self._model.predict_proba(X)