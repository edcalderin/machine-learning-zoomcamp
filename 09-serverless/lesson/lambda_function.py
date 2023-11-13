import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

classes = [
    'dress',
    'hat',
    'longsleeve',
	'outwear',
    'pants',
	'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt']

interpreter = tflite.Interpreter(model_path='clothing_model.tflite')

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

preprocessor = create_preprocessor('xception', (299, 299))

def predict(url: str):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_details[0].get('index'), X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0].get('index'))
    predictions = sorted(
        zip(classes, pred[0].tolist()), 
        key=lambda x:-x[1])
    return predictions

def lambda_handler(event, context):
    url: str = event['url']

    prediction = dict(predict(url))
    return {'prediction': prediction}