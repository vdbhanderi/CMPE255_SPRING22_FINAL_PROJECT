import json
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


class textCorrectionModel:
    def __init__(self) -> None:
        self.max_length = 74
        self.model = None
        self._loadModel()

    def _loadModel(self) -> None:
        self.model = keras.models.load_model(
            'ml/char_level_accuracy_89_3lstm_rnn')

    def deTokenize(self, tokenizer, prediction):
        index_to_words = {id: word for word,
                          id in tokenizer.word_index.items()}
        index_to_words[0] = ''
        pre_index = np.argmax(prediction, 1)
        return ''.join(index_to_words[word] for word in pre_index)

    def predict(self, test_string):
        test_string = [test_string]
        with open('ml/g_text_tokenizer.json') as tokenizer:
            tokenizer_data = json.load(tokenizer)
            g_text_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
                tokenizer_data)

        test_sample = g_text_tokenizer.texts_to_sequences(test_string)
        test_sample = pad_sequences(
            test_sample, maxlen=74, padding='post')

        prediction = self.model.predict(test_sample)
        with open('ml/text_tokenizer.json') as tokenizer:
            tokenizer_data = json.load(tokenizer)
            text_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
                tokenizer_data)
        detoken_prediction = self.deTokenize(text_tokenizer, prediction[0])

        return detoken_prediction
