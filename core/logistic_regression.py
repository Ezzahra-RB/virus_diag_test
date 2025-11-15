from sklearn.linear_model import LogisticRegression
from core.model import Model
from sklearn.preprocessing import StandardScaler

class LogisticRegressionModel(Model):
    def __init__(self):
        self.model = LogisticRegression(max_iter=2000)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path="model.pkl"):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path="model.pkl"):
        import joblib
        self.model = joblib.load(path)
