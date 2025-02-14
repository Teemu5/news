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
from utils import build_and_load_weights  # Adjust import as needed
import numpy as np
from sklearn.linear_model import LogisticRegression

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
# For ensemble, we assume multiple Fastformer models (here simulated by dummy functions)
# In practice, you might load three separate models with build_and_load_weights
model1 = None
model2 = None
model3 = None

# Load configuration and models during startup
@app.on_event("startup")
def load_model_data():
    global user_profiles, tfidf_matrix, news_df, fastformer_user_profiles, model1, model2, model3
    try:
        config = BertConfig.from_json_file('fastformer.json')
        from models import Model  # Ensure Model is in your PYTHONPATH
        # Load three models for ensemble
        model1 = Model(config)
        model1.load_state_dict(torch.load('/app/downloads/fastformer_model1.pth', map_location=torch.device('cpu')))
        model1.eval()

        model2 = Model(config)
        model2.load_state_dict(torch.load('/app/downloads/fastformer_model2.pth', map_location=torch.device('cpu')))
        model2.eval()

        model3 = Model(config)
        model3.load_state_dict(torch.load('/app/downloads/fastformer_model3.pth', map_location=torch.device('cpu')))
        model3.eval()

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

# Pydantic model for request input
class RecommendationRequest(BaseModel):
    user_id: str
    input_text: str

# --- Dummy Prediction Functions for Ensemble Methods ---
# In practice, these functions should call model1, model2, and model3 for inference.
def fastformer_model1_predict(input_text: str) -> np.ndarray:
    # For example, perform preprocessing and model inference using model1
    # Here, we simulate with dummy predictions
    return np.array([0.80, 0.55, 0.30, 0.20])

def fastformer_model2_predict(input_text: str) -> np.ndarray:
    return np.array([0.75, 0.60, 0.35, 0.25])

def fastformer_model3_predict(input_text: str) -> np.ndarray:
    return np.array([0.85, 0.50, 0.25, 0.15])

# --- Ensemble Methods Implementation ---

# Bagging: simple averaging
def ensemble_bagging(input_text: str) -> np.ndarray:
    y1 = fastformer_model1_predict(input_text)
    y2 = fastformer_model2_predict(input_text)
    y3 = fastformer_model3_predict(input_text)
    predictions = np.vstack([y1, y2, y3])
    return np.mean(predictions, axis=0)

# Boosting: weighted averaging (weights inversely proportional to errors)
def ensemble_boosting(input_text: str, errors: np.ndarray) -> np.ndarray:
    y1 = fastformer_model1_predict(input_text)
    y2 = fastformer_model2_predict(input_text)
    y3 = fastformer_model3_predict(input_text)
    predictions = np.vstack([y1, y2, y3])
    errors = np.where(errors == 0, 1e-6, errors)
    weights = 1 / errors
    weights = weights / np.sum(weights)
    return np.average(predictions, axis=0, weights=weights)

# Stacking: meta-model learns to combine base predictions
def train_stacking_meta_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    meta_model = LogisticRegression()
    meta_model.fit(X_train, y_train)
    return meta_model

def ensemble_stacking(input_text: str, meta_model: LogisticRegression) -> np.ndarray:
    y1 = fastformer_model1_predict(input_text)
    y2 = fastformer_model2_predict(input_text)
    y3 = fastformer_model3_predict(input_text)
    X = np.vstack([y1, y2, y3]).T  # each column is a model's prediction
    final_predictions = meta_model.predict_proba(X)[:, 1]
    return final_predictions

# Hybrid Ensemble: combine outputs from bagging, boosting, and stacking
def hybrid_ensemble(input_text: str, boosting_errors: np.ndarray, stacking_meta_model: LogisticRegression) -> np.ndarray:
    bagging_pred = ensemble_bagging(input_text)
    boosting_pred = ensemble_boosting(input_text, boosting_errors)
    stacking_pred = ensemble_stacking(input_text, stacking_meta_model)
    # Final ensemble: simple average of the three ensemble outputs
    final_prediction = (bagging_pred + boosting_pred + stacking_pred) / 3
    return final_prediction

# --- Recommendations Endpoint with Ensemble Integration ---
@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str = Path(..., description="The unique identifier of the user"),
    method: str = Query("tfidf", description="Recommendation method: 'tfidf', 'fastformer', or 'ensemble'"),
    ensemble_method: str = Query(None, description="For ensemble method: choose 'bagging', 'boosting', 'stacking', or 'hybrid'")
):
    if tfidf_matrix is None or user_profiles is None or news_df is None:
        raise HTTPException(status_code=500, detail="Model data not loaded")
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    try:
        if method.lower() == "ensemble":
            if ensemble_method is None:
                raise HTTPException(status_code=400, detail="Ensemble method not specified")
            # For demonstration, use dummy ensemble predictions based on input_text
            if ensemble_method.lower() == "bagging":
                final_scores = ensemble_bagging("dummy input: " + user_id)
            elif ensemble_method.lower() == "boosting":
                dummy_errors = np.array([0.2, 0.15, 0.25])
                final_scores = ensemble_boosting("dummy input: " + user_id, dummy_errors)
            elif ensemble_method.lower() == "stacking":
                X_train_dummy = np.array([
                    [0.80, 0.75, 0.85],
                    [0.55, 0.60, 0.50],
                    [0.30, 0.35, 0.25],
                    [0.20, 0.25, 0.15]
                ])
                y_train_dummy = np.array([1, 0, 1, 0])
                meta_model = train_stacking_meta_model(X_train_dummy, y_train_dummy)
                final_scores = ensemble_stacking("dummy input: " + user_id, meta_model)
            elif ensemble_method.lower() == "hybrid":
                dummy_errors = np.array([0.2, 0.15, 0.25])
                X_train_dummy = np.array([
                    [0.80, 0.75, 0.85],
                    [0.55, 0.60, 0.50],
                    [0.30, 0.35, 0.25],
                    [0.20, 0.25, 0.15]
                ])
                y_train_dummy = np.array([1, 0, 1, 0])
                meta_model = train_stacking_meta_model(X_train_dummy, y_train_dummy)
                final_scores = hybrid_ensemble("dummy input: " + user_id, dummy_errors, meta_model)
            else:
                raise HTTPException(status_code=400, detail="Invalid ensemble method specified")
            
            # For demonstration, assume recommended_indices are derived from final_scores:
            top_n = 5
            recommended_indices = np.argsort(final_scores)[-top_n:][::-1]
        
        elif method.lower() == 'fastformer':
            if user_id not in fastformer_user_profiles:
                raise HTTPException(status_code=404, detail="Fastformer user profile not found")
            user_vector = np.asarray(fastformer_user_profiles[user_id])
            if user_vector.ndim == 1:
                user_vector = user_vector.reshape(1, -1)
            log_ids = torch.LongTensor([user_vector]).to('cpu')
            dummy_targets = torch.zeros(log_ids.size(0)).long().to('cpu')
            with torch.no_grad():
                predictions = model(log_ids, dummy_targets, "")
            if isinstance(predictions, tuple):
                predictions = predictions[1]
            top_n = 5
            recommended_indices = torch.topk(predictions, k=top_n).indices.cpu().numpy().flatten()
        else:
            # tfidf-based recommendations
            user_vector = np.asarray(user_profiles[user_id])
            if user_vector.ndim == 1:
                user_vector = user_vector.reshape(1, -1)
            similarities = cosine_similarity(user_vector, tfidf_matrix)
            recommended_indices = similarities.argsort()[0][-5:][::-1]
        
        # Fetch article details for recommended indices
        recommended_articles = news_df.iloc[recommended_indices][['Title', 'Abstract']].to_dict(orient='records')
        return recommended_articles

    except Exception as e:
        print(f"Error processing recommendations for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
