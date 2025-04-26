from sklearn.linear_model import LogisticRegression
from .base import BaseTitanicEstimator


class TitanicLogisticRegressionEstimator(BaseTitanicEstimator):
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs', random_state=42):
        super().__init__(random_state=random_state)
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self._model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            random_state=random_state,
            max_iter=1000
        )

    def fit(self, X, y):
        super().fit(X, y)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X):
        super().predict_proba(X)  # Checks if fitted
        return self._model.predict_proba(X)