import requests

from ml.model import textCorrectionModel

url = 'http://localhost:4000/results'
r = requests.post(url, json={'sentence': "hello my name is uolive"})

print(r.json())
