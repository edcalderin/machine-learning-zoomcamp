import requests
import json

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://6nv7u5n4sj.execute-api.us-east-2.amazonaws.com/clothes-api/predict'

event = {'url': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJs8cWy35WqJWtYd1EJREiTdT3_qkP33-bTA&usqp=CAU'}

response = requests.post(url, json=event).json()

print(json.dumps(response, indent=4))
