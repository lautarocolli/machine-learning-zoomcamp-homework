from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"