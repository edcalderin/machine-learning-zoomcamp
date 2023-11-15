from fastapi import FastAPI, HTTPException
from tf_serving_connect import TFServingPrediction
import uvicorn

app = FastAPI()

@app.post('/predict')
def predict(data: dict):
    try:
        if 'image_url' in data:
            prediction = TFServingPrediction.from_url(data['image_url'])
        elif 'image' in data:
            prediction = TFServingPrediction.from_path(data['image'])
        else:
            raise HTTPException(status_code=404, detail="Item not found")
    except HTTPException:
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        return prediction.predict()

if __name__=='__main__':
    uvicorn.run(app, port=8080)
    