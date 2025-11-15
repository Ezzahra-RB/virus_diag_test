from pipeline.trainer import train_model
from pipeline.evaluate import evaluate
from core.logistic_regression import LogisticRegressionModel

if __name__ == "__main__":
    model, X_test, y_test = train_model()
    evaluate(model, X_test, y_test)

    # Exemple d'utilisation avec ClinicalPredictor
    from clinical_predictor import ClinicalPredictor
    predictor = ClinicalPredictor(model.model)

    sample = X_test.iloc[0].values  # premier patient du test
    result = predictor.diagnose(sample)
    print("Résultat du diagnostic :", result)
