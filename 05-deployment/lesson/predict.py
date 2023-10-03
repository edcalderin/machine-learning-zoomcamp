import pickle
from flask import Flask, request, jsonify

C = 1.0
model_file = f'model_C={C}.pkl'

app = Flask('ChurnPrediction')

# Load the model
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.post('/predict')
def predict():
    customer = request.get_json()
    X = dv.transform(customer)
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    response = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    print(response)
    return jsonify(response)

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)    