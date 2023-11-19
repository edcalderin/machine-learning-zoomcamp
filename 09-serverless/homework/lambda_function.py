import tflite_runtime.interpreter as tflite
from io import BytesIO
import numpy as np
from PIL import Image
from urllib import request

interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url: str):
    X = download_image(url)
    X = prepare_image(X, (150, 150))
    X = np.array(X, dtype='float32')
    X = X/255
    X = np.expand_dims(X, 0)
    
    interpreter.set_tensor(input_details[0].get('index'), X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0].get('index'))
    return pred[0][0].tolist()

def lambda_handler(event, context):
    url: str = event['url']

    prediction = predict(url)
    return {'prediction': prediction}