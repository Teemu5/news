from flask import Flask, jsonify, request
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import torch
from models import Model
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfOutput, BertIntermediate, BertOutput
config=BertConfig.from_json_file('fastformer.json')
model = Model(config)
model.load_state_dict(torch.load('/app/downloads/fastformer_model.pth', map_location=torch.device('cpu')))
model.eval()
#import torch.optim as optim
#optimizer = optim.Adam([ {'params': model.parameters(), 'lr': 1e-3}])
#model.cuda()
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Recommendations at /recommendations/<userId>"


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=str(e)), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify(error="An internal error occurred, please try again."), 500

import os
import time

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "up"}), 200

# Check if the model is finished processing
def check_model_ready():
    while not os.path.exists('/app/data/_READY'):
        print("sleeping 5 secs")
        app.logger.info("sleeping 5 secs")
        time.sleep(5)
    app.logger.info("end sleep")
    print("end sleep")

#check_model_ready()

def format_articles_for_mind(articles):
    formatted_articles = []
    for article in articles:
        formatted_article = {
            'NewsID': article['url'],  # URL as a unique identifier
            'Category': 'General',  # GNews does not provide category; set default or infer if possible
            'SubCategory': 'None',  # No subcategory information available
            'Title': article['title'],
            'Abstract': article['description'] if 'description' in article else '',
            'URL': article['url'],
            'TitleEntities': '',  # Placeholder as GNews does not provide entity information
            'AbstractEntities': ''  # Placeholder
        }
        formatted_articles.append(formatted_article)
    return formatted_articles

from gnews import GNews

def fetch_news_with_gnews(keywords='latest news'):
    google_news = GNews(language='en', country='US', period='1d', max_results=10)
    articles = google_news.get_news(keywords)
    
    formatted_articles = format_articles_for_mind(articles)
    return formatted_articles

# Initialize the variables at the top
user_profiles, tfidf_matrix, news_df = {}, None, None
fastformer_user_profiles = {}
# Attempt to load precomputed data
try:
    with open('/app/data/user_profiles.pkl', 'rb') as f:
        user_profiles = pickle.load(f)
    with open('/app/data/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('/app/data/news_df.pkl', 'rb') as f:  # Assuming news details are saved in this file
        news_df = pickle.load(f)
    with open('/app/data/fastformer_user_profiles.pkl', 'rb') as f:
        fastformer_user_profiles = pickle.load(f)
except Exception as e:
    app.logger.error(f"Failed to load models: {e}")
    raise
def preprocess_input_ids(input_ids, valid_range): # Why are there negatives?
    # Replace negative values or out-of-range values with padding index (0)
    input_ids = torch.clamp(input_ids, min=0, max=valid_range-1)
    return input_ids

@app.route('/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    method = request.args.get('method', 'tfidf').lower()
    if tfidf_matrix is None or user_profiles is None or news_df is None:
        app.logger.error("Model data not loaded or is incomplete.")
        return jsonify({"error": "Model data not loaded"}), 500

    if user_id not in user_profiles:
        return jsonify({"error": "User profile not found"}), 404
    error_msg=""
    try:
        user_vector = user_profiles[user_id]
        user_vector = np.asarray(user_vector)  # Convert to ndarray if it's not already
        print(f"user_id:{user_id}")
        print(f"user_profiles[user_id]:{user_profiles[user_id]}")
        error_msg += f" user_id:{user_id}"
        if user_vector.ndim == 1:
            user_vector = user_vector.reshape(1, -1)
        if method == 'fastformer':
            user_vector = fastformer_user_profiles[user_id]
            print(f"fastformer_user_profiles['U72015']:{fastformer_user_profiles['U72015']}")
            print(f"fastformer_user_profiles[user_id]:{fastformer_user_profiles[user_id]}")
            user_vector = np.asarray(user_vector)
            print(f"user_vector:{user_vector}")
            error_msg += f"user_vector:{user_vector}"
            log_ids = torch.LongTensor([user_vector]).to('cpu')  # Assuming inputs are tokenized
            print(f"log_ids:{log_ids}")
            # Create a dummy target tensor (size should match your batch size)
            dummy_targets = torch.zeros(log_ids.size(0)).long().to('cpu')
            print(f"dummy_targets:{dummy_targets}")

            # Run the model to get predictions
            with torch.no_grad():
                predictions = model(log_ids, dummy_targets, error_msg)
            if isinstance(predictions, str):
                error_msg = predictions
                raise ValueError("Variable is a string, raising exception.")
            print(f"predictions:{predictions}")
            error_msg += f"predictions:{predictions}"
            if isinstance(predictions, tuple):
                predictions = predictions[1]  # Get the predictions tensor
            else:
                predictions = predictions
            # Get top N recommendations based on predictions
            top_n = 5
            recommended_indices = torch.topk(predictions, k=top_n).indices.cpu().numpy().flatten()
        else:
            similarities = cosine_similarity(user_vector, tfidf_matrix)
            recommended_indices = similarities.argsort()[0][-5:][::-1]

        # Fetch the details of the recommended articles
        recommended_articles = news_df.iloc[recommended_indices][['Title', 'Abstract']].to_dict(orient='records')
        return jsonify(recommended_articles)
    except Exception as e:
        app.logger.error(f"Error processing recommendations for {user_id}: {e}")
        return jsonify({"error": str(e), "error_msg": error_msg}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
