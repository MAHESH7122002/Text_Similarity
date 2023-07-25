import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import requests
import json
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

import torch
from transformers import BertTokenizer, BertModel

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

# def cosine_similarity_bert(text1,text2):
#   return cosine_similarity(get_sentence_embeding([text1]),get_sentence_embeding([text2]))[0][0]

def get_sentence_embeddings(sentence):
    # Tokenize the text and convert it to input tensors
    sentence = re.sub(r'[^a-zA-Z0-9]', ' ', sentence)
    print(type(sentence))
    inputs = tokenizer(sentence, return_tensors='pt',padding=True, truncation=True)
 
    # Forward pass through the BERT model
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the word embeddings from the last layer of BERT
    word_embeddings = outputs.last_hidden_state
    # Extract the embeddings for the first token (CLS token)
    cls_embedding = word_embeddings[:, 0, :]
    # Convert the embeddings to a numpy array
    cls_embedding = cls_embedding.numpy()
    return cls_embedding

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/explore')
def explore():
    return render_template('main.html')

@app.route('/request_api')
def request_api():
    return render_template('request_api.html')


@app.route('/predict',methods = ['POST'])
def predict():
    text =  [str(x) for x in request.form.values()]
    text1 = text[0]
    text2 = text[1]
    embedding1 = get_sentence_embeddings(text1)
    embedding2 = get_sentence_embeddings(text2)
    # prediction = cosine_similarity_word2vec(text1,text2)
    prediction = cosine_similarity(embedding1,embedding2)[0][0]

    print(prediction)
    return render_template('main.html', prediction_text="Similarity Score: {}".format(prediction))

@app.route('/show_table',methods = ['POST'])
def show_table():
    with open('output.json', 'r') as f:
        json_data = json.load(f)
    url = 'http://127.0.0.1:5000/predict_api'
    
    data = json_data[:2]
    response = requests.post(url, json=data)
    output = list(response.json().values())
    prediction = [data,output]
    return render_template('request_api.html', prediction_text=prediction)


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data_json = list(request.get_json(force=True))

    result=[]
    print(data_json)
    for i in range(2):
        print(type(data_json[i]))
        text1 = data_json[i]['text1']
        text2 = data_json[i]['text2']

        embedding1 = get_sentence_embeddings(text1)
        embedding2 = get_sentence_embeddings(text2)
        prediction = cosine_similarity(embedding1,embedding2)[0][0]
        output = round(prediction,2)
        output  = float(prediction)
        result.append(output)
    return jsonify({'similarity_score':result})



if __name__ == '__main__':
    app.run(debug=True)