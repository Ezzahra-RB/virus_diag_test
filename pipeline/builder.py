from abc import ABC, abstractmethod
from pipeline.ml_pipeline import MLPipeline

class PipelineBuilder(ABC):

    def __init__(self):
        self.pipeline = MLPipeline()

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def split(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    def build(self):
        return self.pipeline
