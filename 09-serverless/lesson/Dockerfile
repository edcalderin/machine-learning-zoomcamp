FROM public.ecr.aws/lambda/python:3.11

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp311-cp311-linux_x86_64.whl

RUN pip install keras-image-helper tflite-runtime

COPY lambda_function.py clothing_model.tflite ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]