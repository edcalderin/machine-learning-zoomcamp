FROM public.ecr.aws/lambda/python:3.9

RUN pip install keras-image-helper tflite-runtime

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl?raw=true

COPY lambda_function.py clothing_model.tflite ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]