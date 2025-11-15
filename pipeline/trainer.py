from core.logistic_regression import LogisticRegressionModel
from core.dataset import load_dataset
from sklearn.model_selection import train_test_split

def train_model():
    X, y = load_dataset("data/disease_diagnosis.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    model.save("model.pkl")

    return model, X_test, y_test
