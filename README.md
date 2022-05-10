# CMPE255_SPRING22_FINAL_PROJECT

Team Name - Team Globus

Group Members -

- Jack Zhu (JJLZ21)
- Virag Bhanderi
- Afroz Inamdar
- Alireza Mahin parvar
- Kanupriya Sanjay Agrawal

https://www.kaggle.com/datasets/samarthagarwal23/spelling-mistake-data-1mn

## Requirement and Version

Using device that has GPU memory is greater than 8GB

pipenv ==2022.1.8
tensorflow ==2.8.0
numpy ==1.21.6
Flask ==2.1.2
flask-ngrok ==0.0.25

## Instruction

```
pip install pipenv==2022.1.8
pipenv shell
pip install -r requirement.txt
python app.py
```

## Issue

### Model training

Always run out of GPU Memory due to RNN's Total params is huge
![plot](./picture/out_of_GPU_memory.png)

## Model

Using char level vocabulary for text correction.
40 vocabulary in grammared texts (a-z 0-9 ' )
39 vocabulary in corrected texts

![plot](/picture/vocabuary_graph.png)

Trying the first experiement model using Embedding and Bidrectional LSTM (Long Short Term Memory)
![plot](\picture\bid_rnn_model.png)

Next trying the second experiement model using Embedding with seq2seq (encoding layer and decoding layer)
![plot](./picture/emb_lstm_seq2seq_model.png)

Final model is combining both experiemnts above for better model developing
![plot](./picture/bid_and_emb_seq2seq_model.png)

Final accuracy in traning: 0.9246
Final accuracy in validation: 0.8994
![plot](./picture/test_dataset_evaluation.png)

## Reference

"Towards better decoding and language model integration in sequence to sequence models" (Jan Chorowski, Navdeep Jaitly) https://arxiv.org/abs/1612.02695

"Bidirectional recurrent neural networks" (M. Schuster, K.K. Paliwal) https://ieeexplore.ieee.org/abstract/document/650093
