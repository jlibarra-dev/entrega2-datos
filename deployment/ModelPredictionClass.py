from joblib import load
import numpy as np

class PredictionModel:

    def __init__(self):
        self.model = load("final_model/finalized_model.sav")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result

array = [np.zeros(43)]
prediction = PredictionModel().make_predictions(array)

print(prediction)