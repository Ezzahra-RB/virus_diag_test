from pipeline.clinical_builder import ClinicalPipelineBuilder
from pipeline.trainer import TrainerDirector
from pipeline.evaluate import evaluate

from clinical_predictor import ClinicalPredictor

if __name__ == "__main__":
    # Construction du pipeline via Builder + Director
    builder = ClinicalPipelineBuilder()
    director = TrainerDirector(builder)

    pipeline = director.construct_pipeline()
    model, X_test, y_test = pipeline.run()

    # Évaluation du modèle
    evaluate(model, X_test, y_test)

    # Exemple d'utilisation avec ClinicalPredictor
    predictor = ClinicalPredictor(model.model)

    sample = X_test.iloc[0].values
    result = predictor.diagnose(sample)
    print("Résultat du diagnostic :", result)








# #from pipeline.trainer import train_model
# from pipeline.evaluate import evaluate
# # from core.logistic_regression import LogisticRegressionModel
# from pipeline.clinical_builder import ClinicalPipelineBuilder
# from pipeline.trainer import TrainerDirector

# if __name__ == "__main__":
#     builder = ClinicalPipelineBuilder()
#     director = TrainerDirector(builder)
#     pipeline = director.construct_pipeline()
#     model, X_test, y_test = pipeline.run()
#     #model, X_test, y_test = train_model()
#     evaluate(model, X_test, y_test)

#     # Exemple d'utilisation avec ClinicalPredictor
#     from clinical_predictor import ClinicalPredictor
#     predictor = ClinicalPredictor(model.model)

#     sample = X_test.iloc[0].values  # premier patient du test
#     result = predictor.diagnose(sample)
#     print("Résultat du diagnostic :", result)
