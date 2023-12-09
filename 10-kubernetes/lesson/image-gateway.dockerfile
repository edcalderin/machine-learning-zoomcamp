FROM python:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY predict_api.py proto.py tf_serving_connect.py ./

EXPOSE 8080

ENTRYPOINT ["uvicorn", "--host=0.0.0.0", "--port=8080", "predict_api:app"]