# Steps to reproduce this lesson

1. Run:

```bash
pip install pipenv
pipenv install
pipenv shell
python train.py
```

The last step will create a binary file containing the model with the name `model_C=1.0.pkl`


2. Start docker:
```bash
docker build -t zoomcamp-test .
docker run -it --rm -p 9696:9696 zoomcamp-test:latest
```

3. Open a new terminal, and run:
```python test_predict.py```