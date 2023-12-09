import requests

URL = 'http://localhost:9696/predict'
URL = 'http://a5542d48d79dc4f1fb8661c016ce32cc-1045424854.us-east-2.elb.amazonaws.com/predict'
data = {
        'image_url': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJs8cWy35WqJWtYd1EJREiTdT3_qkP33-bTA&usqp=CAU'
    }
response = requests.post(URL, json=data).json()
print(response)