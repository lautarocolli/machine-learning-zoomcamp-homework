import requests

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://yoz3wv2w08.execute-api.us-west-2.amazonaws.com/test/predict'
print(url)

request = {
    "url": "http://bit.ly/mlbookcamp-pants"
}

result = requests.post(url, json=request).json()
print(result)