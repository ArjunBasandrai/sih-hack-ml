import pandas as pd
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    sentence_embedding = torch.mean(embeddings, dim=1)
    return sentence_embedding.cpu().squeeze().numpy()
try:
    data = pd.read_pickle('disease_data_with_embeddings.pkl')
except FileNotFoundError:
    data = pd.read_csv('disease_data.csv')
    data['Symptoms'] = data['Symptoms'].str.lower()
    data['Disease'] = data['Disease'].str.lower()
    data['Embedding'] = data['Symptoms'].apply(get_embedding)
    data.to_pickle('disease_data_with_embeddings.pkl')

def predict_disease(user_input):
    user_embedding = get_embedding(user_input.lower())
    embeddings = np.stack(data['Embedding'].values)
    similarities = cosine_similarity([user_embedding], embeddings)[0]
    closest_match_index = similarities.argmax()
    disease = data.iloc[closest_match_index]['Disease']
    solution = data.iloc[closest_match_index]['Solution']
    return disease, solution

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json.get('symptoms')
    disease, solution = predict_disease(symptoms)
    return jsonify({
        'predicted_disease': disease,
        'solution': solution
    })

if __name__ == '__main__':
    app.run(debug=True)
