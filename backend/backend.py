from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import os
import torch
import pandas as pd
import time
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
    method: str = Query("tfidf", description="Recommendation method: 'tfidf', 'fastformer', or 'ensemble'")
):
    ensemble_method = method
    global user_profiles, tfidf_matrix, news_df, fastformer_user_profiles, pt_model, model1, model2, model3
    if tfidf_matrix is None or user_profiles is None or news_df is None:
        raise HTTPException(status_code=500, detail="Model data not loaded")
    
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    try:
        if method.lower() in ["bagging", "boosting", "stacking", "hybrid"]:
            if ensemble_method is None:
                raise HTTPException(status_code=400, detail="Ensemble method not specified")
            # Use a dummy input string based on the user_id for ensemble predictions.
            input_text = "dummy input: " + user_id
            history_tensor, candidate_tensor = generate_input_tensors_for_user(user_id, news_df, behaviors_df, tokenizer, max_history_length=50, max_title_length=30)
            if ensemble_method.lower() == "bagging":
                final_scores = ensemble_bagging(history_tensor, candidate_tensor, models_dict)
            elif ensemble_method.lower() == "boosting":
                dummy_errors = np.array([0.2, 0.15, 0.25])
                final_scores = ensemble_boosting(history_tensor, candidate_tensor, models_dict, dummy_errors)
            elif ensemble_method.lower() == "stacking":
                X_train_dummy = np.array([[0.80, 0.75, 0.85],
                                        [0.55, 0.60, 0.50],
                                        [0.30, 0.35, 0.25],
                                        [0.20, 0.25, 0.15]])
                y_train_dummy = np.array([1, 0, 1, 0])
                meta_model = train_stacking_meta_model(X_train_dummy, y_train_dummy)
                final_scores = ensemble_stacking(history_tensor, candidate_tensor, models_dict, meta_model)
            elif ensemble_method.lower() == "hybrid":
                dummy_errors = np.array([0.2, 0.15, 0.25])
                X_train_dummy = np.array([[0.80, 0.75, 0.85],
                                        [0.55, 0.60, 0.50],
                                        [0.30, 0.35, 0.25],
                                        [0.20, 0.25, 0.15]])
                y_train_dummy = np.array([1, 0, 1, 0])
                meta_model = train_stacking_meta_model(X_train_dummy, y_train_dummy)
                final_scores = hybrid_ensemble(history_tensor, candidate_tensor, models_dict, dummy_errors, meta_model)
            else:
                raise HTTPException(status_code=400, detail="Invalid ensemble method specified")
            
            top_n = 5
            recommended_indices = np.argsort(final_scores)[-top_n:][::-1]

        elif method.lower() == "fastformer":
            if user_id not in fastformer_user_profiles:
                raise HTTPException(status_code=404, detail="Fastformer user profile not found")
            # Use the PyTorch model for fastformer-based recommendations.
            user_vector = fastformer_user_profiles[user_id]
            user_vector = np.asarray(user_vector)
            log_ids = torch.LongTensor([user_vector]).to('cpu')
            dummy_targets = torch.zeros(log_ids.size(0), dtype=torch.long).to('cpu')
            error_msg = ""
            with torch.no_grad():
                predictions = pt_model(log_ids, dummy_targets, error_msg)
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
        
        # Retrieve article details from news_df
        recommended_articles = news_df.iloc[recommended_indices][['Title', 'Abstract']].to_dict(orient='records')
        return recommended_articles

    except Exception as e:
        print(f"Error processing recommendations for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
