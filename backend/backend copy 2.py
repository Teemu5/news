from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import os
import time
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertConfig
from models import Model  # Your custom model definition

app = FastAPI(title="News Recommendation API")

# Configure CORS (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
user_profiles = {}
tfidf_matrix = None
news_df = None
fastformer_user_profiles = {}
model = None

# Load the PyTorch model and other data during startup
@app.on_event("startup")
def load_model_data():
    global user_profiles, tfidf_matrix, news_df, fastformer_user_profiles, model
    try:
        # Load the PyTorch model
        config = BertConfig.from_json_file('fastformer.json')
        model = Model(config)
        model.load_state_dict(torch.load('/app/downloads/fastformer_model.pth', map_location=torch.device('cpu')))
        model.eval()
        
        # Load pickled data
        with open('/app/data/user_profiles.pkl', 'rb') as f:
            user_profiles = pickle.load(f)
        with open('/app/data/tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open('/app/data/news_df.pkl', 'rb') as f:
            news_df = pickle.load(f)
        with open('/app/data/fastformer_user_profiles.pkl', 'rb') as f:
            fastformer_user_profiles = pickle.load(f)
            
    except Exception as e:
        print(f"Failed to load model data: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "Recommendations available at /recommendations/<user_id>"}

@app.get("/health")
async def health_check():
    return {"status": "up"}

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str = Path(..., description="The unique identifier of the user"),
    method: str = Query("tfidf", description="Recommendation method: 'tfidf' or 'fastformer'")
):
    global user_profiles, tfidf_matrix, news_df, fastformer_user_profiles, model
    if tfidf_matrix is None or user_profiles is None or news_df is None:
        raise HTTPException(status_code=500, detail="Model data not loaded")
    
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    try:
        if method.lower() == "fastformer":
            if user_id not in fastformer_user_profiles:
                raise HTTPException(status_code=404, detail="Fastformer user profile not found")
            # Retrieve the fastformer user vector and prepare input for the model
            user_vector = fastformer_user_profiles[user_id]
            user_vector = np.asarray(user_vector)
            log_ids = torch.LongTensor([user_vector]).to('cpu')
            dummy_targets = torch.zeros(log_ids.size(0), dtype=torch.long).to('cpu')
            error_msg = ""
            with torch.no_grad():
                predictions = model(log_ids, dummy_targets, error_msg)
            if isinstance(predictions, tuple):
                predictions = predictions[1]
            # Get top 5 recommendations using torch.topk
            top_n = 5
            recommended_indices = torch.topk(predictions, k=top_n).indices.cpu().numpy().flatten()
        else:
            # tfidf-based recommendations
            user_vector = user_profiles[user_id]
            user_vector = np.asarray(user_vector)
            if user_vector.ndim == 1:
                user_vector = user_vector.reshape(1, -1)
            similarities = cosine_similarity(user_vector, tfidf_matrix)
            recommended_indices = similarities.argsort()[0][-5:][::-1]
        
        # Fetch the recommended articles from news_df
        recommended_articles = news_df.iloc[recommended_indices][['Title', 'Abstract']].to_dict(orient='records')
        return recommended_articles
    except Exception as e:
        print(f"Error processing recommendations for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
