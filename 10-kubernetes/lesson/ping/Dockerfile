FROM python:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY ping_api.py ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "--host=0.0.0.0", "--port=9696", "ping_api:app"]