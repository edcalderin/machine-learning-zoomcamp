import pickle
import os
from flask import Flask, request

app = Flask('05-homework')

MODEL_NAME = os.getenv('MODEL_NAME', 'model1.bin')

with open('dv.bin', 'rb') as pkl_file:
    dv = pickle.load(pkl_file)

with open(MODEL_NAME, 'rb') as pkl_file:
    model = pickle.load(pkl_file)
    
@app.post('/predict')
def predict():
    customer = request.get_json()
    
    X = dv.transform(customer)

    prediction = round(model.predict_proba(X)[0, 1], 3)
    
    print(prediction)
    
    return {
        'probability': float(prediction),
        'prediction': bool(prediction > 0.5)
    }
    