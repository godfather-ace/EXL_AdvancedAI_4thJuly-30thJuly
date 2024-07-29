import requests

from app import IrisData

data = IrisData(sepal_length=5.1,
                sepal_width=3.5,
                petal_length=1.4,
                petal_width=0.2)

response = requests.post("http://127.0.0.1:49580/predict", json=data.dict())

print(response.json())
