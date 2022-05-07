from tokenize import String
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import warnings
warnings.filterwarnings('ignore')
import pickle

app = Flask(__name__)
run_with_ngrok(app)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    string_features = [String(x) for x in request.form.values()]
    final_features = [np.array(string_features)]
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Correct sentence should be $ {}'.format(prediction))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    # output = prediction[0]
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)