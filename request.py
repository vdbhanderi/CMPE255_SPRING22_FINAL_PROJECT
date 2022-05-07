import requests

from ml.model import textCorrectionModel
"""
Summary:

just call the class and fill in the input_string

import: 

from ml.model import textCorrectionModel

running:

input_string = "wadeye residents eskeing return to outstaitosn"
model = textCorrectionModel()
print("model prediction: \n", model.predict(input_string))

Returns:
    _type_: String
"""

# TODO: just change the input_string to other string
input_string = "wadeye residents eskeing return to outstaitosn"
model = textCorrectionModel()
print("model prediction: \n", model.predict(input_string))


url = 'http://localhost:4000/results'
# r = requests.post(url,json={'rate':5, 'sales_in_first_month':200, 'sales_in_second_month':400})
r = requests.post(url, json={'sentence': "hello my name is uolive"})

print(r.json())
