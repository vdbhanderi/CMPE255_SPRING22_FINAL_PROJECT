from ml.model import textCorrectionModel
import pickle
from tokenize import String
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
run_with_ngrok(app)

input_string = "wadeye residents eskeing return to outstaitosn"
model = textCorrectionModel()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    string_features = [x for x in request.form.values()]
    # print("inside predict -----string features", string_features)
    prediction = model.predict(string_features[0])
    # print("inside predict -----prediction", prediction)
    return render_template('index.html', prediction_text='Correct sentence should be: {}'.format(prediction))


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    # print("inside results -----data", data)
    prediction = model.predict([np.array(list(data.values()))])
    # print("inside results -----prediction", prediction)
    # print("inside results -----jsonify", jsonify(prediction))
    return jsonify(prediction)


if __name__ == "__main__":
    app.debug = True
    app.run()
