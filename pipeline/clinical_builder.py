import pandas as pd
from pipeline.builder import PipelineBuilder
from utils.preprocessing import preprocess_data
from core.logistic_regression import LogisticRegressionModel
from sklearn.model_selection import train_test_split

class ClinicalPipelineBuilder(PipelineBuilder):

    def load_data(self):
        self.pipeline.df = pd.read_csv("data/disease_diagnosis.csv")
        return self

    def preprocess(self):
        X, y = preprocess_data(self.pipeline.df)
        self.pipeline.X = X
        self.pipeline.y = y
        return self

    def split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.pipeline.X,
            self.pipeline.y,
            test_size=0.2,
            random_state=42
        )
        self.pipeline.X_train = X_train
        self.pipeline.X_test = X_test
        self.pipeline.y_train = y_train
        self.pipeline.y_test = y_test
        return self

    def build_model(self):
        self.pipeline.model = LogisticRegressionModel()
        return self
