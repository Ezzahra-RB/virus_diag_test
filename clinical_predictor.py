class ClinicalPredictor:
    def __init__(self, model):
        self.model = model

    def diagnose(self, patient_data):
        prediction = self.model.predict([patient_data])
        return "infecté" if prediction[0] == 1 else "sain"
