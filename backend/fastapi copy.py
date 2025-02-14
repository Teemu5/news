from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import time
from models import Model
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfOutput, BertIntermediate, BertOutput
from gnews import GNews
import pandas as pd
from fastformer_clusters import build_and_load_weights

# Initialize FastAPI app
app = FastAPI(title="News Recommendation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model data
user_profiles = {}
tfidf_matrix = None
news_df = None
fastformer_user_profiles = {}
model = None

# Load the configuration and model during startup
@app.on_event("startup")
def load_model_data():
    global user_profiles, tfidf_matrix, news_df, fastformer_user_profiles, model
    try:
        config = BertConfig.from_json_file('fastformer.json')
        from models import Model  # Ensure Model is in your PYTHONPATH
        model = Model(config)
        model.load_state_dict(torch.load('/app/downloads/fastformer_model.pth', map_location=torch.device('cpu')))
        model.eval()

        with open('/app/data/user_profiles.pkl', 'rb') as f:
            user_profiles = pickle.load(f)
        with open('/app/data/tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open('/app/data/news_df.pkl', 'rb') as f:
            news_df = pickle.load(f)
        with open('/app/data/fastformer_user_profiles.pkl', 'rb') as f:
            fastformer_user_profiles = pickle.load(f)
    except Exception as e:
        # Use standard logging if needed
        print(f"Failed to load model data: {e}")
        raise e

# Define Pydantic model for request input if needed
class RecommendationRequest(BaseModel):
    user_id: str
    method: str = "tfidf"  # default recommendation method

# Home endpoint
@app.get("/")
async def home():
    return {"message": "Recommendations available at /recommendations/{user_id}"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "up"}

# Utility function: Format articles for output
def format_articles_for_mind(articles):
    formatted_articles = []
    for article in articles:
        formatted_article = {
            'NewsID': article['url'],
            'Category': 'General',
            'SubCategory': 'None',
            'Title': article['title'],
            'Abstract': article.get('description', ''),
            'URL': article['url'],
            'TitleEntities': '',
            'AbstractEntities': ''
        }
        formatted_articles.append(formatted_article)
    return formatted_articles

# Function to fetch news using GNews
def fetch_news_with_gnews(keywords='latest news'):
    google_news = GNews(language='en', country='US', period='1d', max_results=10)
    articles = google_news.get_news(keywords)
    formatted_articles = format_articles_for_mind(articles)
    return formatted_articles

# Preprocess input_ids function (unchanged)
def preprocess_input_ids(input_ids, valid_range):
    input_ids = torch.clamp(input_ids, min=0, max=valid_range-1)
    return input_ids

# Recommendations endpoint: GET /recommendations/{user_id}
@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str = Path(..., description="The unique identifier of the user"),
    method: str = Query("tfidf", description="Recommendation method: 'tfidf' or 'fastformer'")
):
    # Check if required data is loaded
    if tfidf_matrix is None or user_profiles is None or news_df is None:
        raise HTTPException(status_code=500, detail="Model data not loaded")

    # Check if user profile exists
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="User profile not found")

    error_msg = ""
    try:
        if method.lower() == 'fastformer':
            # Use fastformer profiles
            if user_id not in fastformer_user_profiles:
                raise HTTPException(status_code=404, detail="Fastformer user profile not found")
            user_vector = fastformer_user_profiles[user_id]
            user_vector = np.asarray(user_vector)
            error_msg += f" user_id:{user_id}"
            if user_vector.ndim == 1:
                user_vector = user_vector.reshape(1, -1)
            error_msg += f" user_vector:{user_vector}"
            log_ids = torch.LongTensor([user_vector]).to('cpu')
            dummy_targets = torch.zeros(log_ids.size(0)).long().to('cpu')
            with torch.no_grad():
                predictions = model(log_ids, dummy_targets, error_msg)
            if isinstance(predictions, str):
                error_msg = predictions
                raise ValueError("Model returned an error string")
            if isinstance(predictions, tuple):
                predictions = predictions[1]  # use the second output if tuple
            top_n = 5
            recommended_indices = torch.topk(predictions, k=top_n).indices.cpu().numpy().flatten()
        else:
            # Use tfidf-based recommendation
            user_vector = user_profiles[user_id]
            user_vector = np.asarray(user_vector)
            if user_vector.ndim == 1:
                user_vector = user_vector.reshape(1, -1)
            similarities = cosine_similarity(user_vector, tfidf_matrix)
            recommended_indices = similarities.argsort()[0][-5:][::-1]

        # Fetch recommended articles details
        recommended_articles = news_df.iloc[recommended_indices][['Title', 'Abstract']].to_dict(orient='records')
        return recommended_articles

    except Exception as e:
        # Log error message (use print or proper logging)
        print(f"Error processing recommendations for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
