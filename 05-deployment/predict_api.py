import pickle
import logging
from datetime import datetime

from flask import Flask
from flask import request
from flask import jsonify

# Set up logging
logging.basicConfig(filename='requests.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    # Log the request
    logging.info(f"Prediction request: {customer}")

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    # Log the result
    logging.info(f"Prediction result: {result}")

    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

# Example customer data:
# {
#     "gender": "female",
#     "seniorcitizen": 0,
#     "partner": "yes",
#     "dependents": "no",
#     "phoneservice": "no",
#     "multiplelines": "no_phone_service",
#     "internetservice": "dsl",
#     "onlinesecurity": "no",
#     "onlinebackup": "yes",
#     "deviceprotection": "no",
#     "techsupport": "no",
#     "streamingtv": "no",
#     "streamingmovies": "no",
#     "contract": "month-to-month",
#     "paperlessbilling": "yes",
#     "paymentmethod": "electronic_check",
#     "tenure": 1,
#     "monthlycharges": 29.85,
#     "totalcharges": 29.85
# }