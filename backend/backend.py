from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import os
import torch
import pandas as pd
import time
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertConfig
from models import Model  # Your PyTorch model definition
from utils import build_and_load_weights, get_models, generate_input_tensors_for_user  # Helper that builds & loads a Keras model
from recommender import (  # import functions from recommender module
    fastformer_model_predict,
    ensemble_bagging,
    ensemble_boosting,
    train_stacking_meta_model,
    ensemble_stacking,
    hybrid_ensemble
)
from dateutil.parser import isoparse

app = FastAPI(title="News Recommendation API")

# Configure CORS (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data and models
user_profiles = {}
tfidf_matrix = None
fastformer_user_profiles = {}

# PyTorch fastformer model (for fastformer method)
pt_model = None

# ---------------------------
# Dummy Preprocessing Function
# ---------------------------
def tokenize_input(input_text: str):
    """
    Dummy preprocessing to transform raw input text into the 
    required input shape for Keras ensemble models.
    Replace this with your actual text preprocessing.
    """
    return np.array([[len(input_text), 1]])

# ---------------------------
# Startup: Load Models and Data
# ---------------------------
@app.on_event("startup")
def load_model_data():
    global user_profiles, tfidf_matrix, fastformer_user_profiles, news_df, behaviors_df, tokenizer, models_dict
    global pt_model, model1, model2, model3
    try:
        # Load PyTorch fastformer model
        config = BertConfig.from_json_file('fastformer.json')
        pt_model = Model(config)
        pt_model.load_state_dict(torch.load('/app/downloads/fastformer_model.pth', map_location=torch.device('cpu')))
        pt_model.eval()

        # Load Keras ensemble models.
        #model1 = build_and_load_weights('/app/models/fastformer_cluster_0_full_balanced_1_epoch.weights.h5')
        #model2 = build_and_load_weights('/app/models/fastformer_cluster_1_full_balanced_1_epoch.weights.h5')
        #model3 = build_and_load_weights('/app/models/fastformer_cluster_2_full_balanced_1_epoch.weights.h5')
        # OR TRAIN MODELS
        models_dict, news_df, behaviors_df, tokenizer = get_models()
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

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Recommendations available at /recommendations/<user_id>"}

@app.get("/health")
async def health_check():
    return {"status": "up"}
@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str = Path(..., description="The unique identifier of the user"),
    method: str = Query("tfidf", description="Recommendation method: 'tfidf', 'fastformer', or 'ensemble'"),
    ref_date: str = Query(None, description="Optional reference date in ISO format (e.g. 2023-02-01T00:00:00)"),
    max_candidates: int = Query(-1, description="Optional maximum number of candidate articles to consider")
):
    global user_profiles, tfidf_matrix, news_df, fastformer_user_profiles, pt_model, models_dict
    if tfidf_matrix is None or user_profiles is None or news_df is None:
        raise HTTPException(status_code=500, detail="Model data not loaded")
    
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    try:
        # Parse the reference date if provided
        current_date = None
        if ref_date:
            try:
                #current_date = datetime.fromisoformat(ref_date)
                current_date = isoparse(ref_date)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid date format (ref_date={ref_date}). Use ISO format (e.g. 2023-02-01T00:00:00)")
        
        # For ensemble methods, generate input tensors for all candidate articles based on the reference date and max_candidates.
        if method.lower() in ["bagging", "boosting", "stacking", "hybrid"]:
            history_tensor, candidate_tensors, candidate_ids = generate_input_tensors_for_user(
                user_id, news_df, behaviors_df, tokenizer,
                max_history_length=50, max_title_length=30,
                candidate_timeframe_hours=24,
                current_date=current_date,
                max_candidates=max_candidates
            )
            candidate_scores = []
            candidate_debug_info = []
            for idx, candidate_tensor in enumerate(candidate_tensors):
                cand_id = candidate_ids[idx]
                cand_title_arr = news_df.loc[news_df['NewsID'] == cand_id, 'Title'].values
                cand_title = cand_title_arr[0] if len(cand_title_arr) > 0 else "Unknown Title"
                if method.lower() == "bagging":
                    score = ensemble_bagging(history_tensor, candidate_tensor, models_dict)
                elif method.lower() == "boosting":
                    dummy_errors = np.array([0.2, 0.15, 0.25])
                    score = ensemble_boosting(history_tensor, candidate_tensor, models_dict, dummy_errors)
                elif method.lower() == "stacking":
                    X_train_dummy = np.array([
                        [0.80, 0.75, 0.85],
                        [0.55, 0.60, 0.50],
                        [0.30, 0.35, 0.25],
                        [0.20, 0.25, 0.15]
                    ])
                    y_train_dummy = np.array([1, 0, 1, 0])
                    meta_model = train_stacking_meta_model(X_train_dummy, y_train_dummy)
                    score = ensemble_stacking(history_tensor, candidate_tensor, models_dict, meta_model)
                elif method.lower() == "hybrid":
                    dummy_errors = np.array([0.2, 0.15, 0.25])
                    X_train_dummy = np.array([
                        [0.80, 0.75, 0.85],
                        [0.55, 0.60, 0.50],
                        [0.30, 0.35, 0.25],
                        [0.20, 0.25, 0.15]
                    ])
                    y_train_dummy = np.array([1, 0, 1, 0])
                    meta_model = train_stacking_meta_model(X_train_dummy, y_train_dummy)
                    score = hybrid_ensemble(history_tensor, candidate_tensor, models_dict, dummy_errors, meta_model)
                else:
                    raise HTTPException(status_code=400, detail="Invalid ensemble method specified")
                candidate_scores.append(score)
                candidate_debug_info.append({
                    "NewsID": cand_id,
                    "Title": cand_title,
                    "Score": score
                })
                print(f"Candidate {idx}: NewsID={cand_id}, Title='{cand_title}', Score={score}")
            
            candidate_scores = np.array(candidate_scores).flatten()
            top_n = 5
            recommended_indices = np.argsort(candidate_scores)[-top_n:][::-1]
            print("\nAll candidate scores:")
            for info in candidate_debug_info:
                print(info)
            print("\nTop candidate indices:", recommended_indices)
        
        elif method.lower() == "fastformer":
            if user_id not in fastformer_user_profiles:
                raise HTTPException(status_code=404, detail="Fastformer user profile not found")
            # Use the PyTorch model for fastformer-based recommendations.
            user_vector = fastformer_user_profiles[user_id]
            user_vector = np.asarray(user_vector)
            log_ids = torch.LongTensor([user_vector]).to('cpu')
            dummy_targets = torch.zeros(log_ids.size(0), dtype=torch.long).to('cpu')
            with torch.no_grad():
                predictions = pt_model(log_ids, dummy_targets, "")
            if isinstance(predictions, tuple):
                predictions = predictions[1]
            top_n = 5
            recommended_indices = torch.topk(predictions, k=top_n).indices.cpu().numpy().flatten()
        else:
            # TFIDF-based recommendations
            user_vector = user_profiles[user_id]
            user_vector = np.asarray(user_vector)
            if user_vector.ndim == 1:
                user_vector = user_vector.reshape(1, -1)
            similarities = cosine_similarity(user_vector, tfidf_matrix)
            recommended_indices = similarities.argsort()[0][-5:][::-1]
        
        # Retrieve article details from news_df using recommended_indices.
        # Here, the recommended_indices from ensemble methods correspond to the index within candidate_ids.
        if method.lower() in ["bagging", "boosting", "stacking", "hybrid"]:
            recommended_ids = [candidate_ids[i] for i in recommended_indices]
            recommended_articles = news_df[news_df['NewsID'].isin(recommended_ids)][['Title', 'Abstract']].to_dict(orient='records')
        else:
            recommended_articles = news_df.iloc[recommended_indices][['Title', 'Abstract']].to_dict(orient='records')
        
        return recommended_articles

    except Exception as e:
        print(f"Error processing recommendations for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
