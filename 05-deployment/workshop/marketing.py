import requests

url = 'http://localhost:8080/predict'

customer = {
  "gender": "male",
  "seniorcitizen": 0,
  "partner": "no",
  "dependents": "yes",
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "tenure": 6,
  "monthlycharges": 29.85,
  "totalcharges": 129.85,
}

response = requests.post(url, json= customer)
response_json = response.json()
churn = response_json['churn_probability']

if churn >= 0.5:
    print(f'Churn: YES (prob of churning: {churn:.3f})')
else:
    print(f'Churn: NO (prob of churning: {churn:.3f})')