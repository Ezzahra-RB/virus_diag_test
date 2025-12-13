class MLPipeline:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def run(self):
        self.model.train(self.X_train, self.y_train)
        self.model.save("model.pkl")
        return self.model, self.X_test, self.y_test
