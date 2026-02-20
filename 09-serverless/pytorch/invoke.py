import boto3
import json

lambda_client = boto3.client('lambda', region_name='us-west-2')

payload = {
    "url": "http://bit.ly/mlbookcamp-pants"
}

response = lambda_client.invoke(
    FunctionName='clothing-classifier',
    InvocationType='RequestResponse',
    Payload=json.dumps(payload)
)

result = json.loads(response['Payload'].read())
print(json.dumps(result, indent=2))