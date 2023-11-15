from keras_image_helper import create_preprocessor
from proto import make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import grpc
import os

HOST = os.getenv('TF_SERVING_HOST', 'localhost:8500')

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

class TFServingPrediction:
    
    def __init__(self, data) -> None:
        self.__data = data
    
    @classmethod
    def from_url(cls, url: str):
        preprocessor = create_preprocessor('xception', target_size=(299,299))
        data = preprocessor.from_url(url)
        return cls(data)
    
    @classmethod
    def from_path(cls, path):
        preprocessor = create_preprocessor('xception', target_size=(299,299))
        data = preprocessor.from_path(path)
        return cls(data)
    
    def __np_to_protobuf(self):
        if self.__data.dtype!= 'float32':
            self.__data = self.__data.astype('float32')
        return make_tensor_proto(self.__data)

    def __prepare_response(self, pb_response):
        predictions = pb_response.outputs['dense_19'].float_val
        return dict(sorted(zip(classes, predictions), key=lambda x: -x[1]))
    
    def predict(self):
        X = self.__np_to_protobuf()
        channel = grpc.insecure_channel(HOST)

        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        pb_request = predict_pb2.PredictRequest()
        pb_request.model_spec.name = 'clothing-model'
        pb_request.model_spec.signature_name = 'serving_default'
        print('here')
        pb_request.inputs['input_20'].CopyFrom(X)

        pb_response = stub.Predict(pb_request, timeout=20.0)
        
        return self.__prepare_response(pb_response)