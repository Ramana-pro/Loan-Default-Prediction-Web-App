import joblib
import numpy as np

def predict_default(input_data):
    model = joblib.load("models/best_model.pkl")
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return "Default" if prediction[0] == 1 else "No Default"
