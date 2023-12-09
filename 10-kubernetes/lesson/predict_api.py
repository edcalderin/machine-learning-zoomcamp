from fastapi import FastAPI, HTTPException
from tf_serving_connect import TFServingPrediction
app = FastAPI()

@app.post('/predict')
def predict(data: dict):
    try:
        if 'image_url' in data:
            prediction = TFServingPrediction.from_url(data['image_url'])
        elif 'image' in data:
            prediction = TFServingPrediction.from_path(data['image'])
        else:
            raise HTTPException(status_code=404, detail='Item not found')
    except HTTPException:
        raise HTTPException(status_code=404, detail='Item not found')
    else:
        return prediction.predict()

if __name__ == '__main__':
    data = {
        'image_url': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJs8cWy35WqJWtYd1EJREiTdT3_qkP33-bTA&usqp=CAU'
    }
    print(predict(data))
    