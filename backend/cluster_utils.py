import os
import re
import time
import nltk
import pandas as pd
import numpy as np
import zipfile
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dot, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pickle
import tensorflow as tf
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta
from dateutil.parser import isoparse
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommender import (  # import functions from recommender module
    fastformer_model_predict,
    ensemble_bagging,
    ensemble_boosting,
    train_stacking_meta_model,
    ensemble_stacking,
    hybrid_ensemble
)
import hashlib
from traceback import format_exc
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# (Assume your other functions – candidate_generation, candidate_scoring, etc. – are defined above.)
# --- [Cleaning Function] ---
def clean_text(text):
    if pd.isna(text):
        return ''
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    words = text.split()
    words = [w for w in words if not w in stop_words]
    return ' '.join(words)
import logging
import datetime

date_str = datetime.datetime.now().strftime("%Y-%m-%d")
log_filename = f"cluster_profile_{date_str}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# ===================== Cluster-Level Evaluation Functions =====================
def build_cluster_profile(cluster_users, behaviors_df, news_df, cutoff_time, tokenizer, max_history_length=50, max_title_length=30):
    """
    Build an aggregated cluster profile based on user histories.
    This function processes each user's history and aggregates them.
    
    Logs progress every 100 users.
    """
    cluster_history_tensor = []
    cluster_profile_ids = []
    
    # Optionally, open a file to append progress (if you prefer manual file writes)
    # with open("cluster_profile_progress.log", "a") as log_file:
    
    total_users = len(cluster_users)
    for i, user_id in enumerate(cluster_users):
        # Filter the user history by time (assuming cutoff_time is provided as a datetime)
        try:
            behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'])

            # Convert cutoff_time to a datetime
            cutoff_time = pd.to_datetime(cutoff_time)
            user_hist = behaviors_df[
                (behaviors_df['UserID'] == user_id) & 
                (behaviors_df['Time'] <= np.datetime64(cutoff_time))
            ]
        except Exception as e:
            logging.error(f"Error processing user {user_id}: {e}")
            continue

        # Process each user's history (this is a placeholder for your actual processing)
        if not user_hist.empty:
            # For example, get history text from the latest row:
            latest_sample = user_hist.iloc[-1]
            history_text = latest_sample.get("HistoryText", "")
            # Tokenize and pad history_text, for instance:
            history_sequence = tokenizer.texts_to_sequences([history_text])
            history_padded = pad_sequences(history_sequence, maxlen=max_title_length, padding='post', truncating='post', value=0)
            cluster_history_tensor.append(history_padded)
            cluster_profile_ids.append(user_id)
        
        # Log progress every 100 users
        if (i + 1) % 1000 == 0:
            logging.info(f"Processed {i + 1} of {total_users} users in build_cluster_profile")
    
    # Log final progress
    logging.info(f"Finished processing all {total_users} users for cluster profile")
    
    # Aggregate the user histories into a cluster profile tensor, for example by averaging:
    if cluster_history_tensor:
        cluster_profile_tensor = np.mean(np.array(cluster_history_tensor), axis=0)
    else:
        cluster_profile_tensor = None

    return cluster_profile_tensor, cluster_profile_ids
def build_cluster_profile2(cluster_users, behaviors_df, news_df, cutoff_time, tokenizer, max_history_length=50, max_title_length=30):
    """
    Build a cluster history by aggregating (union) the articles clicked by each user in the cluster 
    *before* the cutoff_time. Then, create a pseudo-history tensor using the news titles.
    Returns:
      - cluster_history_tensor: a tensor of shape (1, max_history_length, max_title_length)
      - cluster_profile_ids: list of article IDs in the cluster history (aggregated)
    """
    # For each user, take their history (articles with interactions <= cutoff_time)
    cluster_profile_set = set()
    # Also, store a list of (article_id, interaction_time) so that we can sort by time.
    article_times = {}
    cutoff_time = pd.to_datetime(cutoff_time)
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    total_users = len(cluster_users)
    for i, user_id in enumerate(cluster_users):
        user_hist = behaviors_df[(behaviors_df['UserID'] == user_id) & (behaviors_df['Time'] <= np.datetime64(cutoff_time))]
        for idx, row in user_hist.iterrows():
            # Parse history from HistoryText (assume space‐separated article IDs)
            if pd.isna(row["HistoryText"]) or row["HistoryText"].strip() == "":
                continue
            for art in row["HistoryText"].split():
                cluster_profile_set.add(art)
                # Record the interaction time if not already present or update with a later time
                t = row["Time"]
                if art not in article_times or t > article_times[art]:
                    article_times[art] = t
                # Log progress every 100 users
        if (i + 1) % 1000 == 0:
            logging.info(f"Processed {i + 1} of {total_users} users in build_cluster_profile2")

    # Now, sort the profile items by time (we choose the most recent ones)
    sorted_profile = sorted(list(cluster_profile_set), key=lambda art: article_times.get(art, np.datetime64('1970-01-01')), reverse=True)
    # Limit to max_history_length items
    if len(sorted_profile) > max_history_length:
        sorted_profile = sorted_profile[:max_history_length]
    # Build a "cluster history" text by concatenating the titles for each article.
    profile_texts = []
    for art in sorted_profile:
        # Look up the article title (or CombinedText) from news_df
        title_arr = news_df.loc[news_df['NewsID'] == art, 'CombinedText'].values
        if len(title_arr) > 0:
            profile_texts.append(str(title_arr[0]))
    # For tokenization, you may simply join them into a single string.
    cluster_profile_text = " ".join(profile_texts)
    # Now, tokenize this text as a "history" for the cluster.
    cluster_history_seq = tokenizer.texts_to_sequences([cluster_profile_text])
    cluster_history_padded = pad_sequences(cluster_history_seq, maxlen=max_title_length, padding='post', truncating='post', value=0)
    # We want a tensor of shape (1, max_history_length, max_title_length). If we have less than max_history_length rows, pad with zeros.
    if cluster_history_padded.shape[0] < max_history_length:
        pad_rows = np.zeros((max_history_length - cluster_history_padded.shape[0], max_title_length), dtype=int)
        cluster_history_padded = np.vstack([pad_rows, cluster_history_padded])
    else:
        cluster_history_padded = cluster_history_padded[-max_history_length:]
    cluster_history_tensor = tf.convert_to_tensor([cluster_history_padded], dtype=tf.int32)
    return cluster_history_tensor, sorted_profile

def get_cluster_profile_cached(cluster_users, behaviors_df, news_df, cutoff_time, tokenizer, cache_file="cluster_profile.pkl"):
    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"Loading cluster profile from {cache_file}")
        logging.info(f"Loading cluster profile from {cache_file}")
        with open(cache_file, "rb") as f:
            cluster_profile, profile_ids = pickle.load(f)
    else:
        print(f"Building cluster profile {cache_file}...")
        logging.info(f"Building cluster profile {cache_file}...")
        # Run your slow build_cluster_profile function
        cluster_profile, profile_ids = build_cluster_profile2(cluster_users, behaviors_df, news_df, cutoff_time, tokenizer)
        # Save the results to a cache file
        with open(cache_file, "wb") as f:
            pickle.dump((cluster_profile, profile_ids), f)
    return cluster_profile, profile_ids
def get_cache_filename(cluster_users, cutoff_time, cache_dir, base_name="ground_truth_freq", cluster_id=""):
    """
    Create a unique cache filename based on the cutoff time, cluster ID, and the sorted list of cluster users.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Create a unique string based on the sorted cluster users.
    users_str = ",".join(sorted(cluster_users))
    users_hash = hashlib.md5(users_str.encode('utf-8')).hexdigest()
    cutoff_time_str = pd.to_datetime(cutoff_time).strftime("%Y%m%d_%H%M%S")
    cache_filename = os.path.join(cache_dir, f"{base_name}_{cluster_id}_{cutoff_time_str}_{users_hash}.pkl")
    return cache_filename

def load_cache(cache_filename, force_new=False):
    """
    If a cache file exists and force_new is False, load and return the cached object.
    Otherwise, return None.
    """
    if os.path.exists(cache_filename) and not force_new:
        logging.info(f"Loading cached data from {cache_filename}")
        with open(cache_filename, 'rb') as f:
            return pickle.load(f)
    return None
def get_cluster_ground_truth_frequency(cluster_users, behaviors_df, cutoff_time, cache_dir="cache", cluster_id = "", force_new = False):
    """
    Build a frequency-based ground truth: count how many times each article was clicked by users in the cluster after cutoff_time.
    """
    cache_filename = get_cache_filename(cluster_users, cutoff_time, cache_dir, "ground_truth_freq", cluster_id)
    cached_result = load_cache(cache_filename, force_new)
    if cached_result is not None:
        return cached_result

    # Ensure the Time column is in datetime format.
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    cutoff_time_dt = pd.to_datetime(cutoff_time)
    ground_truth_freq = {}
    total_users = len(cluster_users)
    for i, user_id in enumerate(cluster_users):
        user_future = behaviors_df[
            (behaviors_df['UserID'] == user_id) & (behaviors_df['Time'] > np.datetime64(cutoff_time_dt))
        ]
        for _, row in user_future.iterrows():
            if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
                continue
            for imp in row["Impressions"].split():
                try:
                    art, label = imp.split('-')
                    if int(label) == 1:
                        ground_truth_freq[art] = ground_truth_freq.get(art, 0) + 1
                except Exception as e:
                    print(f"Error parsing impression {imp}: {e}")
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1} of {total_users} users in get_cluster_ground_truth_frequency")
    # Save the result in the cache.
    with open(cache_filename, 'wb') as f:
        pickle.dump(ground_truth_freq, f)
    logging.info(f"Cached ground truth frequency to {cache_filename}")
    return ground_truth_freq

def compute_weighted_precision_recall_at_k(recommended_ids, ground_truth_freq, k):
    """
    Compute weighted precision and recall based on click frequency.
    
    recommended_ids: ordered list of recommended article IDs.
    ground_truth_freq: dict mapping article IDs to click counts.
    k: number of top recommendations.
    
    Here, weighted precision is calculated relative to the best possible cumulative frequency
    that could be achieved by recommending the top-k most popular articles.
    """
    recommended_k = recommended_ids[:k]
    weighted_hits = sum(ground_truth_freq.get(rec, 0) for rec in recommended_k)
    total_clicks = sum(ground_truth_freq.values())
    
    # The ideal top-k sum is computed from the k highest frequencies in the ground truth.
    top_k_possible = sum(sorted(ground_truth_freq.values(), reverse=True)[:k])
    
    precision = weighted_hits / top_k_possible if top_k_possible > 0 else 0
    recall = weighted_hits / total_clicks if total_clicks > 0 else 0
    return precision, recall

def cluster_evaluate_weighted(recommended_ids, cluster_ground_truth_freq, k):
    """
    Evaluate a cluster's recommendations using frequency-based weighted precision and recall.
    """
    precision, recall = compute_weighted_precision_recall_at_k(recommended_ids, cluster_ground_truth_freq, k)
    return precision, recall

def get_cluster_ground_truth(cluster_users, behaviors_df, cutoff_time, cache_dir="cache", cluster_id = "", force_new = False):
    """
    Get the union of all articles that any user in the cluster clicked *after* cutoff_time.
    Returns a set of article IDs.
    """
    cache_filename = get_cache_filename(cluster_users, cutoff_time, cache_dir, "ground_truth", cluster_id)
    cached_result = load_cache(cache_filename, force_new)
    if cached_result is not None:
        return cached_result
    
    ground_truth = set()
    cutoff_time = pd.to_datetime(cutoff_time)
    total_users = len(cluster_users)
    for i, user_id in enumerate(cluster_users):
        user_future = behaviors_df[(behaviors_df['UserID'] == user_id) & (behaviors_df['Time'] > np.datetime64(cutoff_time))]
        for idx, row in user_future.iterrows():
            if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
                continue
            for imp in row["Impressions"].split():
                try:
                    art, label = imp.split('-')
                    if int(label) == 1:
                        ground_truth.add(art)
                except Exception as e:
                    print(f"Error parsing impression {imp}: {e}")
        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1} of {total_users} users in get_cluster_ground_truth")
    # Save the result in the cache.
    with open(cache_filename, 'wb') as f:
        pickle.dump(ground_truth, f)
    logging.info(f"Cached ground truth frequency to {cache_filename}")
    return ground_truth

def tfidf_filter_candidates(candidates_df: pd.DataFrame, user_history_text: str, tfidf_vectorizer: TfidfVectorizer, min_similarity: float = 0.1) -> pd.DataFrame:
    """
    Re-rank and optionally filter candidate articles using TF-IDF similarity.
    
    Parameters:
      - candidates_df: DataFrame of candidate articles (should include a text field, e.g. "CombinedText" or "Title").
      - user_history_text: Aggregated text from the user's history.
      - tfidf_vectorizer: A pre-fitted TfidfVectorizer.
      - min_similarity: Minimum cosine similarity threshold for retaining a candidate.
      
    Returns:
      - filtered_df: DataFrame sorted by TF-IDF similarity (descending), possibly filtered by the threshold.
    """
    # Use the candidate text – here we assume you have a "CombinedText" column
    candidate_texts = candidates_df["CombinedText"].tolist()  # or use "Title" if preferred
    candidate_vectors = tfidf_vectorizer.transform(candidate_texts)
    
    # Compute TF-IDF vector for the user's history
    user_vector = tfidf_vectorizer.transform([user_history_text])
    
    # Compute cosine similarities between user history and each candidate
    similarities = cosine_similarity(user_vector, candidate_vectors)[0]

    # Log the similarity distribution (for threshold determination)
    import matplotlib.pyplot as plt
    plt.hist(similarities, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("TF-IDF Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Candidate Similarity Scores")
    plt.savefig("tfidf_similarity_distribution.png")
    plt.close()
    print("TF-IDF similarity distribution plotted and saved as 'tfidf_similarity_distribution.png'.")

    # Add the similarity score to the DataFrame
    candidates_df = candidates_df.copy()
    candidates_df["TFIDF_Similarity"] = similarities
    
    # Optionally filter out candidates below the threshold
    filtered_df = candidates_df[candidates_df["TFIDF_Similarity"] >= min_similarity]
    
    # Sort the remaining candidates by similarity (highest first)
    filtered_df = filtered_df.sort_values(by="TFIDF_Similarity", ascending=False)
    
    return filtered_df

def cluster_candidate_generation(cluster_history_ids, news_df, behaviors_df, cutoff_time, tokenizer, tfidf_vectorizer=None, min_tfidf_similarity=0.02, max_candidates=-1):
    """
    Generate candidate articles for a cluster.
    - Candidate pool: all articles whose first interaction time is <= cutoff_time.
    - Remove any articles already in the cluster history.
    - Optionally apply TF-IDF filtering using the cluster’s aggregated text.
    Returns:
       - candidate_tensors: list of TensorFlow tensors for each candidate article.
       - candidate_ids: list of candidate NewsIDs.
    """

    # First, compute for each article the first interaction time from behaviors_df.
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    behaviors_df['Time'] = behaviors_df['Time'].apply(lambda t: t.tz_convert(None) if (t is not None and t.tzinfo is not None) else t)
    
    # Convert cutoff_time to datetime and remove timezone if present
    cutoff_time_dt = pd.to_datetime(cutoff_time)
    if cutoff_time_dt.tzinfo is not None:
        cutoff_time_dt = cutoff_time_dt.tz_convert(None)
    first_interactions = {}
    # Iterate over behaviors_df to compute the first interaction time for each news article
    for _, row in behaviors_df.iterrows():
        time_val = row["Time"]
        if pd.isna(time_val):
            continue
        if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
            continue
        for imp in row["Impressions"].split():
            try:
                art, label = imp.split('-')
                # Update the first interaction time if this is the earliest seen for the article
                if art not in first_interactions or time_val < first_interactions[art]:
                    first_interactions[art] = time_val
            except Exception as e:
                print(f"Error parsing impression {imp}: {e}")

    # Build candidate pool: articles with first interaction time <= cutoff_time
    candidate_pool = [art for art, t in first_interactions.items() if t <= cutoff_time_dt]
    # Remove articles that are already in the cluster history.
    candidate_pool = list(set(candidate_pool) - set(cluster_history_ids))
    # Build a candidate DataFrame from news_df using NewsID from news_df
    candidates_df = news_df[news_df['NewsID'].isin(candidate_pool)]

    # Optionally apply TF-IDF filtering using the cluster’s aggregated history text.
    if tfidf_vectorizer is not None:
        # Build the cluster history text from the cluster history IDs
        texts = []
        for art in cluster_history_ids:
            title_arr = news_df.loc[news_df['NewsID'] == art, 'CombinedText'].values
            if len(title_arr) > 0:
                texts.append(str(title_arr[0]))
        cluster_history_text = " ".join(texts)
        candidates_df = tfidf_filter_candidates(candidates_df, cluster_history_text, tfidf_vectorizer, min_similarity=min_tfidf_similarity)
    
    # Optionally limit the number of candidates.
    if max_candidates > 0 and len(candidates_df) > max_candidates:
        candidates_df = candidates_df.head(max_candidates)
    
    # For each candidate, create a tensor from its title (or CombinedText)
    candidate_tensors = []
    candidate_ids = []
    for idx, row in candidates_df.iterrows():
        art_id = row['NewsID']
        title = row['Title'] if pd.notna(row['Title']) and row['Title'].strip() != "" else " "
        seq = tokenizer.texts_to_sequences([title])
        if len(seq[0]) == 0:
            seq = [[0]]
        padded = pad_sequences(seq, maxlen=30, padding='post', truncating='post', value=0)[0]
        tensor = tf.convert_to_tensor([padded], dtype=tf.int32)
        candidate_tensors.append(tensor)
        candidate_ids.append(art_id)
    return candidate_tensors, candidate_ids

def cluster_candidate_scoring(cluster_history_tensor, candidate_tensors, models_dict, ensemble_method="bagging"):
    """
    For each candidate in the candidate list, compute a score using the ensemble method.
    (Here we reuse your ensemble_bagging function.)
    Returns a numpy array of scores.
    """
    scores = []
    for candidate_tensor in candidate_tensors:
        score, _ = ensemble_bagging(cluster_history_tensor, candidate_tensor, models_dict)
        scores.append(score)
    scores = np.array(scores).flatten()
    return scores
def cluster_candidate_scoring_single(cluster_history_tensor, candidate_tensors, single_model):
    """
    For each candidate, compute a score using a single Fastformer model.
    Returns a numpy array of scores (same length as candidate_tensors).
    """
    scores = []
    for candidate_tensor in candidate_tensors:
        score = fastformer_model_predict(cluster_history_tensor, candidate_tensor, single_model)
        scores.append(score)
    return np.array(scores).flatten()
def cluster_rank_candidates(candidate_scores, candidate_ids, k):
    """
    Ranks the candidate articles by their score and returns the top-k candidate IDs.
    """
    top_indices = np.argsort(candidate_scores)[-k:][::-1]
    recommended_ids = [candidate_ids[i] for i in top_indices]
    return recommended_ids
def compute_precision_recall_at_k(recommended_ids, ground_truth_ids, k):
    recommended_k = recommended_ids[:k]
    relevant = [1 if rec in ground_truth_ids else 0 for rec in recommended_k]
    precision = sum(relevant) / k
    recall = sum(relevant) / len(ground_truth_ids) if ground_truth_ids else 0
    return precision, recall

def cluster_evaluate(recommended_ids, cluster_ground_truth, k):
    """
    Computes precision@k and recall@k for the cluster.
    """
    # Use the same compute_precision_recall_at_k function
    precision, recall = compute_precision_recall_at_k(recommended_ids, cluster_ground_truth, k)
    return precision, recall
def average_precision_at_k(recommended_ids, ground_truth_ids, k):
    """
    Average Precision for a single user, given top-k recommended_ids and user’s ground_truth_ids.
    """
    recommended_k = recommended_ids[:k]
    hit_count = 0
    sum_precisions = 0.0
    for i, rec in enumerate(recommended_k, start=1):
        if rec in ground_truth_ids:
            hit_count += 1
            sum_precisions += hit_count / i
    if hit_count == 0:
        return 0.0
    return sum_precisions / hit_count

def compute_map_at_k(recommended_ids_list, ground_truth_list, k):
    """
    recommended_ids_list: list of recommended lists, one per user
    ground_truth_list: list of sets/lists of ground_truth IDs, one per user
    returns MAP at k.
    """
    ap_values = []
    for rec_ids, gt_ids in zip(recommended_ids_list, ground_truth_list):
        ap = average_precision_at_k(rec_ids, gt_ids, k)
        ap_values.append(ap)
    return np.mean(ap_values) if ap_values else 0.0

# Similar for nDCG
import math

def dcg_at_k(recommended_ids, ground_truth_ids, k):
    recommended_k = recommended_ids[:k]
    dcg = 0.0
    for i, rec in enumerate(recommended_k, start=1):
        if rec in ground_truth_ids:
            # binary relevance
            dcg += 1.0 / math.log2(i+1)
    return dcg

def compute_ndcg_at_k(recommended_ids_list, ground_truth_list, k):
    ndcg_values = []
    for rec_ids, gt_ids in zip(recommended_ids_list, ground_truth_list):
        idcg = dcg_at_k(list(gt_ids), gt_ids, min(k, len(gt_ids)))  # ideal DCG if we recommended all relevant items first
        if idcg == 0:
            ndcg_values.append(0.0)
            continue
        dcg_val = dcg_at_k(rec_ids, gt_ids, k)
        ndcg_values.append(dcg_val / idcg)
    return np.mean(ndcg_values) if ndcg_values else 0.0

def evaluate_cluster_model(
    cluster_history_tensor,
    candidate_tensors,
    candidate_ids,
    ground_truth,
    ground_truth_freq,
    model_obj,
    model_name,
    k_values,
    cluster_id,
    user_list
):
    """
    Scores candidates using either the ensemble dict or a single model,
    ranks them, and computes evaluation metrics (precision, recall, weighted).
    
    Returns a dictionary of metrics.
    """
    # 1) Score candidates
    if isinstance(model_obj, dict):
        # Ensemble with model dict
        candidate_scores = cluster_candidate_scoring(cluster_history_tensor, candidate_tensors, model_obj)
    else:
        # Single-model scoring
        candidate_scores = cluster_candidate_scoring_single(cluster_history_tensor, candidate_tensors, model_obj)
    metrics = []
    metrics.append({
            "model": model_name,
            "cluster_id": cluster_id,
            "num_users": len(user_list),
            "num_candidates": len(candidate_ids)
        })
    for k in k_values:
        # 2) Rank candidates
        recommended_ids = cluster_rank_candidates(candidate_scores, candidate_ids, k)

        # 3) Evaluate
        precision_weighted, recall_weighted = cluster_evaluate_weighted(recommended_ids, ground_truth_freq, k)
        precision, recall = cluster_evaluate(recommended_ids, ground_truth, k)

        # 4) Add to metrics dict
        metrics[0][f"recommended_ids@{k}"] = recommended_ids
        metrics[0][f"precision@{k}"] = precision
        metrics[0][f"recall@{k}"] = recall
        metrics[0][f"precision_weighted@{k}"] = precision_weighted
        metrics[0][f"recall_weighted@{k}"] = recall_weighted
    return metrics
def run_cluster_experiments(cluster_mapping, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer,
                            cutoff_time_str, k_values=[5,10,20,50,100]):
    """
    cluster_mapping: dict mapping cluster_id -> list of user_ids.
    cutoff_time_str: a string cutoff time (e.g. "2023-02-01T00:00:00Z")
    k_values: number of recommendations per cluster.
    
    For each cluster:
      - Build the cluster profile (history tensor and article IDs).
      - Generate candidate articles.
      - Score candidates.
      - Rank candidates.
      - Compute evaluation metrics (precision and recall) using the cluster ground truth (union of future clicks).
    
    Returns:
       A DataFrame with metrics per cluster.
    """
    cutoff_time = isoparse(cutoff_time_str)
    metrics = []
    for cluster_id, user_list in cluster_mapping.items():
        print(f"\n=== Evaluating Cluster {cluster_id} with {len(user_list)} users ===")
        # Build cluster history profile.
        cluster_history_tensor, cluster_profile_ids = get_cluster_profile_cached(user_list, behaviors_df, news_df, cutoff_time, tokenizer, f"cluster_profile2_{cluster_id}.pkl")
        #cluster_history_tensor, cluster_profile_ids = build_cluster_profile(user_list, behaviors_df, news_df, cutoff_time, tokenizer)
        # Get cluster ground truth (future clicked articles by any member, after cutoff).
        # Example usage within your cluster experiment:
        # Instead of:
        # cluster_ground_truth = get_cluster_ground_truth(user_list, behaviors_df, cutoff_time)
        # Use:
        cluster_ground_truth_freq = get_cluster_ground_truth_frequency(user_list, behaviors_df, cutoff_time,"cache", cluster_id, False)
        cluster_ground_truth = get_cluster_ground_truth(user_list, behaviors_df, cutoff_time,"cache", cluster_id, False)
        print(f"Cluster {cluster_id} ground truth (future clicked): {cluster_ground_truth}")
        # Generate candidate articles for the cluster.
        candidate_tensors, candidate_ids = cluster_candidate_generation(cluster_profile_ids, news_df, behaviors_df, cutoff_time, tokenizer, tfidf_vectorizer)
        print(f"Cluster {cluster_id}: {len(candidate_ids)} candidate articles generated.")

        # Evaluate ensemble
        ensemble_metrics = evaluate_cluster_model(
            cluster_history_tensor=cluster_history_tensor,
            candidate_tensors=candidate_tensors,
            candidate_ids=candidate_ids,
            ground_truth=cluster_ground_truth,
            ground_truth_freq=cluster_ground_truth_freq,
            model_obj=models_dict,              # the dictionary for ensemble
            model_name="ensemble_bagging",
            k_values=k_values,
            cluster_id=cluster_id,
            user_list=user_list
        )
        metrics.extend(ensemble_metrics)

        # Evaluate individual models
        for i, (model_key, single_model) in enumerate(models_dict.items()):
            single_metrics = evaluate_cluster_model(
                cluster_history_tensor=cluster_history_tensor,
                candidate_tensors=candidate_tensors,
                candidate_ids=candidate_ids,
                ground_truth=cluster_ground_truth,
                ground_truth_freq=cluster_ground_truth_freq,
                model_obj=single_model,           # a single model, not a dict
                model_name=f"fastformer_{model_key}",
                k_values=k_values,
                cluster_id=cluster_id,
                user_list=user_list
            )
            metrics.extend(single_metrics)

        
    results_df = pd.DataFrame(metrics)
    results_df.to_csv("cluster_experiment_results.csv", index=False)
    print("Cluster-level experiment results saved to 'cluster_experiment_results.csv'")
    return results_df
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def build_user_profile_tensor(
    user_id,
    behaviors_df,
    news_df,
    cutoff_time,
    tokenizer,
    max_history_length=50,
    max_title_length=30
):
    """
    1) Filter behaviors_df for a single user’s interactions (<= cutoff_time).
    2) Gather all clicked article IDs from 'HistoryText'.
    3) Build a single user-level "history tensor" by tokenizing the text of these articles.
    4) Return (history_tensor, history_article_ids)
    """
    # Ensure 'Time' is datetime
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff_time)

    # Filter to user’s rows up to cutoff
    user_hist = behaviors_df[
        (behaviors_df['UserID'] == user_id)
        & (behaviors_df['Time'] <= np.datetime64(cutoff_dt))
    ].copy()

    # Collect article IDs from user’s HistoryText
    all_hist_article_ids = set()
    for _, row in user_hist.iterrows():
        if pd.isna(row["HistoryText"]) or row["HistoryText"].strip() == "":
            continue
        for art in row["HistoryText"].split():
            all_hist_article_ids.add(art)

    original_history_len = len(all_hist_article_ids)
    # Sort by most recent if you prefer
    # (If you have timestamps for each article in HistoryText, you can do so.
    #  Otherwise, just turn the set to a list.)
    history_article_ids = list(all_hist_article_ids)

    # If there's a time dimension for each article ID, sort by that. Otherwise skip.
    # for example, you can store them in a list with timestamps to sort.
    
    # Keep only up to max_history_length
    if len(history_article_ids) > max_history_length:
        history_article_ids = history_article_ids[-max_history_length:]  # or however you want to slice

    # Build a "history" text from each article’s "CombinedText"
    # or you can build an array of tokenized titles. (Below is an array-of-titles approach.)
    history_titles = []
    for art_id in history_article_ids:
        # Look up the CombinedText from news_df
        rows = news_df[news_df['NewsID'] == art_id]
        if not rows.empty:
            text = str(rows.iloc[0]['CombinedText'])
        else:
            text = ""

        # Tokenize & pad to max_title_length
        seq = tokenizer.texts_to_sequences([text])[0]
        # If empty sequence, put a [0]
        if len(seq) == 0:
            seq = [0]
        seq = pad_sequences([seq], maxlen=max_title_length, padding='post', truncating='post', value=0)[0]
        history_titles.append(seq)
    history_titles = np.array(history_titles, dtype=int)
    if history_titles.ndim == 1 and history_titles.shape[0] == 0:
        history_titles = history_titles.reshape(0, max_title_length)
        logging.info(f"history_titles:{history_titles} empty for user: {user_id}!!!")
    # Now we have a list of shape (num_history, max_title_length). 
    # We must pad/truncate to exactly max_history_length rows:
    if len(history_titles) < max_history_length:
        # pad the top
        padding_needed = max_history_length - len(history_titles)
        padding_rows = np.zeros((padding_needed, max_title_length), dtype=int)
        history_titles = np.vstack([padding_rows, history_titles])
    else:
        history_titles = np.array(history_titles[-max_history_length:])

    # The final shape should be (max_history_length, max_title_length).
    # Convert to a TF tensor if you want:
    history_tensor = tf.convert_to_tensor(history_titles, dtype=tf.int32)

    return history_tensor, history_article_ids, original_history_len
def user_candidate_generation(
    user_id,
    user_history_ids,
    behaviors_df,
    news_df,
    tokenizer,
    tfidf_vectorizer=None,
    cutoff_time=None,
    min_tfidf_similarity=0.02,
    max_candidates=-1,
    max_title_length=30
):
    """
    Build a candidate pool for a single user.
    1) Possibly gather all articles that exist up to cutoff_time (if needed).
    2) Exclude articles in user_history_ids.
    3) Optionally apply TF–IDF filter.
    4) Return candidate_tensors, candidate_ids.
    """

    # Optionally filter articles by cutoff_time
    if cutoff_time is not None:
        behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
        cutoff_dt = pd.to_datetime(cutoff_time)
        if cutoff_dt.tzinfo is not None:
            cutoff_dt = cutoff_dt.tz_convert(None)
        first_interactions = {}
        for _, row in behaviors_df.iterrows():
            if pd.isna(row["Time"]):
                continue
            if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
                continue
            if row["Time"] <= np.datetime64(cutoff_dt):
                # record earliest time
                for imp in row["Impressions"].split():
                    art_id, label = imp.split('-')
                    if art_id not in first_interactions or row["Time"] < first_interactions[art_id]:
                        first_interactions[art_id] = row["Time"]
        candidate_pool = [art for art, t in first_interactions.items() if t <= cutoff_dt]
    else:
        # If no time constraint, let candidate_pool be all news
        candidate_pool = news_df['NewsID'].unique().tolist()

    # remove user's history from the candidate pool
    candidate_pool = list(set(candidate_pool) - set(user_history_ids))

    # Build candidates_df
    candidates_df = news_df[news_df['NewsID'].isin(candidate_pool)].copy()

    # Optional TF–IDF filter
    if tfidf_vectorizer is not None:
        # Build user’s aggregated text from user_history_ids
        texts = []
        for art_id in user_history_ids:
            row_news = news_df[news_df['NewsID'] == art_id]
            if not row_news.empty:
                texts.append(str(row_news.iloc[0]['CombinedText']))
        user_history_text = " ".join(texts)

        # Then your function tfidf_filter_candidates
        candidates_df = tfidf_filter_candidates(candidates_df, user_history_text, tfidf_vectorizer, 
                                                min_similarity=min_tfidf_similarity)

    # Optionally limit the candidate size
    if max_candidates > 0 and len(candidates_df) > max_candidates:
        candidates_df = candidates_df.head(max_candidates)

    # Build the candidate tensors
    candidate_tensors = []
    candidate_ids = []
    for idx, row in candidates_df.iterrows():
        art_id = row['NewsID']
        title = row['Title'] if pd.notna(row['Title']) else ""
        seq = tokenizer.texts_to_sequences([title])[0]
        if len(seq) == 0:
            seq = [0]
        seq = pad_sequences([seq], maxlen=max_title_length, padding='post', truncating='post')[0]
        candidate_tensors.append(tf.convert_to_tensor([seq], dtype=tf.int32))  # shape (1, max_title_length)
        candidate_ids.append(art_id)
    logging.info(f"lens: newsdf:{len(news_df['NewsID'].tolist())}candidate_pool:{len(candidate_pool)},candidate_pool:{len(candidate_pool)}")
    return candidate_tensors, candidate_ids
def get_user_future_clicks(user_id, behaviors_df, cutoff_time):
    """
    Return a set of article IDs that user clicked after cutoff_time.
    Typically from behaviors_df['Impressions'], 
    where each impression is e.g. "N1234-1" meaning article=‘N1234’, label=1
    """
    user_future_clicks = set()
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff_time)

    future_rows = behaviors_df[
        (behaviors_df['UserID'] == user_id)
        & (behaviors_df['Time'] > np.datetime64(cutoff_dt))
    ]
    for _, row in future_rows.iterrows():
        if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
            continue
        for imp in row["Impressions"].split():
            try:
                art_id, label = imp.split('-')
                label = int(label)
                if label == 1:  # means user actually clicked
                    user_future_clicks.add(art_id)
            except:
                pass

    return user_future_clicks
def compute_coverage(all_recommended_ids, total_num_articles):
    """
    all_recommended_ids: a list of lists (each sub-list is recommended IDs for one user)
    total_num_articles: total number of articles in your dataset
    """
    recommended_set = set()
    for rec_list in all_recommended_ids:
        recommended_set.update(rec_list)
    coverage_value = len(recommended_set) / total_num_articles
    return coverage_value
def user_evaluation_loop(models_dict, news_df, behaviors_df, tokenizer, tfidf_vectorizer,
                         user_ids, cutoff_str="2019-11-10T00:00:00Z", chunk_size=1000):
    user_ids = list(user_ids)
    cutoff_dt = pd.to_datetime(cutoff_str)
    
    # For measuring final metrics
    all_metrics = []
    
    for start_idx in range(0, len(user_ids), chunk_size):
        end_idx = start_idx + chunk_size
        chunk_user_ids = user_ids[start_idx:end_idx]
        
        chunk_results = []
        for user_id in chunk_user_ids:
            # 1) build_user_profile_tensor
            user_hist_tensor, user_hist_ids, original_history_len = build_user_profile_tensor(
                user_id, 
                behaviors_df, 
                news_df, 
                cutoff_dt,  # or cutoff_str
                tokenizer
            )
            
            # 2) user_candidate_generation
            candidate_tensors, candidate_ids = user_candidate_generation(
                user_id, 
                user_hist_ids, 
                behaviors_df, 
                news_df, 
                tokenizer, 
                tfidf_vectorizer,
                cutoff_time=cutoff_dt,
                max_candidates=300  # e.g., re-rank 300
            )
            
            # 3) Score (ensemble or single)
                # 1) Score candidates
            if isinstance(models_dict, dict):
                # Ensemble with model dict
                candidate_scores = cluster_candidate_scoring(user_hist_tensor, candidate_tensors, models_dict)
            else:
                # Single-model scoring
                candidate_scores = cluster_candidate_scoring_single(user_hist_tensor, candidate_tensors, models_dict)
            
            # 4) ground truth
            future_clicks = get_user_future_clicks(user_id, behaviors_df, cutoff_dt)
            
            # Example: Evaluate at k=10
            k = 10
            recommended_ids = cluster_rank_candidates(candidate_scores, candidate_ids, k)
            prec, rec = compute_precision_recall_at_k(recommended_ids, future_clicks, k)
            
            # 5) Add to chunk results
            chunk_results.append({
                "user_id": user_id,
                "precision@10": prec,
                "recall@10": rec,
                # etc.
            })
        
        # Save chunk to CSV
        chunk_df = pd.DataFrame(chunk_results)
        chunk_df.to_csv(f"user_metrics_{start_idx}_{end_idx}.csv", index=False)
        
        # If you want global aggregates:
        all_metrics.append(chunk_df)
    
    # Combine all chunks if you want
    final_df = pd.concat(all_metrics, ignore_index=True)
    final_df.to_csv("user_metrics_all.csv", index=False)
    return final_df
def write_results_to_csv(results_list, output_csv="user_level_experiment_results.csv", partial=False):
    """
    Writes a list of dict results to CSV. If partial=True,
    it appends rows (without overwriting or rewriting headers).
    If partial=False, it overwrites the file fully (writes header).
    """
    if not results_list:
        return  # nothing to write

    df = pd.DataFrame(results_list)
    if partial:
        # Append rows with no header if the file already exists:
        mode = "a"
        header = not os.path.exists(output_csv)
    else:
        # Overwrite the file, writing header
        mode = "w"
        header = True

    df.to_csv(output_csv, mode=mode, header=header, index=False)
def write_partial_rows(rows, filename="user_level_partial_results.csv"):
    """
    Appends a list of dictionaries (rows) to a CSV. 
    - If the file doesn't exist, write headers.
    - If it does exist, append without headers.
    """
    if not rows:
        return  # nothing to write

    df = pd.DataFrame(rows)
    file_exists = os.path.exists(filename)

    # If file exists, append without writing header:
    df.to_csv(
        filename,
        mode='a',
        header=not file_exists,  # write header if file doesn't exist
        index=False
    )
def score_candidates_in_batch(history_tensor, candidate_tensors, model, batch_size=128):
    """
    Batch-scoring for a single Keras model:
      - history_tensor: shape (1, max_history_length, max_title_length)
      - candidate_tensors: list of shape (1, max_title_length) for each candidate
      - model: a single Keras model
      - batch_size: the batch size for model.predict()

    Returns a 1D NumPy array of shape (num_candidates,) with predicted scores.
    """

    num_candidates = len(candidate_tensors)
    if num_candidates == 0:
        return np.array([], dtype=float)

    # Stack all candidate rows => shape (num_candidates, max_title_length)
    batch_candidates = tf.concat(candidate_tensors, axis=0)

    # Repeat the user_history_tensor => shape (num_candidates, max_history_length, max_title_length)
    batch_history = tf.repeat(history_tensor, repeats=num_candidates, axis=0)

    # Single predict call
    preds = model.predict(
        {
           "history_input": batch_history,
           "candidate_input": batch_candidates
        },
        batch_size=batch_size
    )
    # preds is shape (num_candidates, 1)
    return preds.ravel()  # flatten to shape (num_candidates,)
def score_candidates_ensemble_batch(history_tensor, candidate_tensors, models_dict, batch_size=128):
    """
    Batch-scoring for an ensemble of models in 'models_dict'.
    We'll do a simple average (like ensemble bagging).
    """
    # We'll store predictions from each model, then average
    all_preds = []
    separate_scores = {}
    for key, model in models_dict.items():
        preds = score_candidates_in_batch(history_tensor, candidate_tensors, model, batch_size)
        all_preds.append(preds)
        separate_scores[key] = preds
    # shape => (num_models, num_candidates)
    # Average across axis=0 => shape (num_candidates,)
    mean_preds = np.mean(all_preds, axis=0)
    return mean_preds, separate_scores
def run_cluster_experiments_user_level(
    cluster_mapping, 
    train_data, 
    test_data,
    news_df,
    behaviors_df,
    models_dict, 
    tokenizer,
    tfidf_vectorizer,
    cutoff_time,
    k_values=[5, 10, 20, 50],
    partial_csv="user_level_partial_results.csv",
    shuffle_clusters=False,       # New parameter: if True, randomize cluster order.
    cluster_order=None            # New parameter: if provided, process clusters in this order.
):
    """
    For each cluster (processed in a specified or random order), for each user:
      - Build user profile from train_data.
      - Generate & score candidates.
      - Evaluate vs. test_data.
    Partial per-user results (including candidate count, number of history articles, 
    and number of future clicked articles for each user) are written to a CSV file incrementally.
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from traceback import format_exc
    logging.info(f"partial_csv: {partial_csv}, cluster_mapping:{cluster_mapping}")
    results = []
    total_articles = len(news_df)
    partial_buffer = []

    # Determine cluster order.
    if cluster_order is not None:
        ordered_clusters = cluster_order
    else:
        ordered_clusters = list(cluster_mapping.keys())
        if shuffle_clusters:
            np.random.shuffle(ordered_clusters)

    for cluster_id in ordered_clusters:
        user_list = cluster_mapping[cluster_id]
        # Initialize accumulators for cluster-level aggregation.
        user_metrics = {k: {"precision": [], "recall": []} for k in k_values}
        user_maps = {k: [] for k in k_values}
        user_ndcgs = {k: [] for k in k_values}
        cluster_recs = []  # For coverage (we use top-10 recommendations)
        total_users = len(user_list)

        for i, user_id in enumerate(tqdm(user_list, desc=f"Evaluating users in cluster {cluster_id}")):
            try:
                logging.info(f"Starting user {user_id} in cluster {cluster_id} (index {i})")
                
                # 1) Build user profile.
                user_history_tensor, user_history_ids, original_history_len = build_user_profile_tensor(
                    user_id, behaviors_df, news_df, cutoff_time, tokenizer
                )
                num_history_articles = len(user_history_ids)
                
                # 2) Build candidates.
                candidate_tensors, candidate_ids = user_candidate_generation(
                    user_id, user_history_ids, train_data, news_df,
                    tokenizer, tfidf_vectorizer, cutoff_time, 0.02
                )
                num_candidates = len(candidate_ids)
                
                # 3) Expand dims (to shape (1, max_history_length, max_title_length))
                user_history_tensor = tf.expand_dims(user_history_tensor, axis=0)
                
                # 4) Score candidates using ensemble.
                candidate_scores, separate_scores = score_candidates_ensemble_batch(
                    user_history_tensor,
                    candidate_tensors,
                    models_dict,
                    batch_size=512
                )
                def evaluate_scores(scores, model_key, partial_csv):
                    # 5) Get ground truth: articles clicked after cutoff.
                    user_future_clicks = get_user_future_clicks(user_id, test_data, cutoff_time)
                    num_future_clicks = len(user_future_clicks)
                    
                    # Prepare partial result row with extra info.
                    partial_result_row = {
                        "cluster_id": cluster_id,
                        "user_id": user_id,
                        "user_index_in_cluster": i,
                        "num_candidates": num_candidates,
                        "num_history_articles": num_history_articles,
                        "original_history_len": original_history_len,
                        "num_future_clicks": num_future_clicks
                    }
                    
                    # For each k, compute evaluation metrics.
                    for k in k_values:
                        recommended_ids = cluster_rank_candidates(candidate_scores, candidate_ids, k)
                        # Record recommendations for coverage using k==10 (as an example)
                        if k == 10:
                            cluster_recs.append(recommended_ids)
                        
                        prec, rec = compute_precision_recall_at_k(recommended_ids, user_future_clicks, k)
                        ap = average_precision_at_k(recommended_ids, user_future_clicks, k)
                        ndcg_val = dcg_at_k(recommended_ids, user_future_clicks, k)
                        ideal_dcg = dcg_at_k(list(user_future_clicks), user_future_clicks, min(k, len(user_future_clicks)))
                        ndcg = ndcg_val / ideal_dcg if ideal_dcg > 0 else 0.0

                        # Accumulate for cluster-level summary.
                        user_metrics[k]["precision"].append(prec)
                        user_metrics[k]["recall"].append(rec)
                        user_maps[k].append(ap)
                        user_ndcgs[k].append(ndcg)
                        
                        # Record per-user metrics.
                        partial_result_row[f"precision_{k}"] = prec
                        partial_result_row[f"recall_{k}"] = rec
                        partial_result_row[f"ap_{k}"] = ap
                        partial_result_row[f"ndcg_{k}"] = ndcg
                        partial_result_row[f"num_recommendations_{k}"] = len(recommended_ids)
                    
                    partial_result_row["status"] = "DONE"
                    partial_buffer.append(partial_result_row)
                    
                    # Flush partial results every 10 users.
                    if (i + 1) % 10 == 0:
                        logging.info(f"Writing partial rows: {partial_buffer}")
                        write_partial_rows(partial_buffer, partial_csv)
                        partial_buffer = []
                evaluate_scores(candidate_scores, "bagging", f"bagging_{partial_csv}")
                for model_key, single_model in models_dict.items():
                    evaluate_scores(separate_scores[model_key], model_key, f"{model_key}_{partial_csv}")
                    
            except Exception as e:
                logging.error(f"Failed on user {user_id} in cluster {cluster_id} with error: {e}\n{format_exc()}")
                partial_buffer.append({
                    "cluster_id": cluster_id,
                    "user_id": user_id,
                    "user_index_in_cluster": i,
                    "status": f"FAILED: {e}"
                })
                continue

        # Flush any remaining rows for this cluster.
        if partial_buffer:
            write_partial_rows(partial_buffer, partial_csv)
            partial_buffer = []
        
        # Summarize cluster-level metrics.
        coverage_cluster = compute_coverage(cluster_recs, total_articles)
        for k in k_values:
            mean_precision = np.mean(user_metrics[k]["precision"]) if user_metrics[k]["precision"] else 0
            mean_recall = np.mean(user_metrics[k]["recall"]) if user_metrics[k]["recall"] else 0
            map_k = np.mean(user_maps[k]) if user_maps[k] else 0
            ndcg_k = np.mean(user_ndcgs[k]) if user_ndcgs[k] else 0

            results.append({
                "cluster_id": cluster_id,
                "k": k,
                "precision_user_level": mean_precision,
                "recall_user_level": mean_recall,
                "MAP": map_k,
                "nDCG": ndcg_k,
                "coverage": coverage_cluster, 
                "num_users": len(user_list)
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("user_level_experiment_results.csv", index=False)
    print("Cluster-level experiment results saved to 'user_level_experiment_results.csv'")
    return results_df


def run_cluster_experiments_user_level2(
    cluster_mapping, 
    train_data, 
    test_data,
    news_df,
    behaviors_df,
    models_dict, 
    tokenizer,
    tfidf_vectorizer,
    cutoff_time,
    k_values=[5, 10, 20, 50],
    partial_csv="user_level_partial_results.csv"
):
    """
    For each cluster, for each user:
      - Build user profile from train_data.
      - Generate & score candidates.
      - Evaluate vs. test_data.
    Partial per-user results (including candidate count and recommendation count for each k)
    are written to a CSV file incrementally.
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from traceback import format_exc

    results = []
    total_articles = len(news_df)
    partial_buffer = []

    for cluster_id, user_list in cluster_mapping.items():
        # Initialize accumulators for cluster-level aggregation.
        user_metrics = {k: {"precision": [], "recall": []} for k in k_values}
        user_maps = {k: [] for k in k_values}
        user_ndcgs = {k: [] for k in k_values}
        cluster_recs = []  # For coverage (we use top-10 recommendations)
        total_users = len(user_list)

        for i, user_id in enumerate(tqdm(user_list, desc=f"Evaluating users in cluster {cluster_id}")):
            try:
                logging.info(f"Starting user {user_id} in cluster {cluster_id} (index {i})")
                
                # 1) Build user profile.
                user_history_tensor, user_history_ids = build_user_profile_tensor(
                    user_id, behaviors_df, news_df, cutoff_time, tokenizer
                )
                
                # 2) Build candidates.
                candidate_tensors, candidate_ids = user_candidate_generation(
                    user_id, user_history_ids, train_data, news_df,
                    tokenizer, tfidf_vectorizer, cutoff_time, 0.02
                )
                num_candidates = len(candidate_ids)
                
                # 3) Expand dims: history tensor shape becomes (1, max_history_length, max_title_length).
                user_history_tensor = tf.expand_dims(user_history_tensor, axis=0)
                
                # 4) Batch score candidates from the ensemble.
                candidate_scores, separate_scores = score_candidates_ensemble_batch(
                    user_history_tensor,
                    candidate_tensors,
                    models_dict,
                    batch_size=512
                )
                
                # 5) Get ground truth.
                user_future_clicks = get_user_future_clicks(user_id, test_data, cutoff_time)
                
                # Prepare partial result row for this user.
                partial_result_row = {
                    "cluster_id": cluster_id,
                    "user_id": user_id,
                    "user_index_in_cluster": i,
                    "num_candidates": num_candidates  # new field: total candidates generated
                }
                
                # For each k, compute metrics and also record number of recommendations.
                for k in k_values:
                    recommended_ids = cluster_rank_candidates(candidate_scores, candidate_ids, k)
                    # Record candidate recommendations for coverage (using k==10, for example)
                    if k == 10:
                        cluster_recs.append(recommended_ids)
                    
                    prec, rec = compute_precision_recall_at_k(recommended_ids, user_future_clicks, k)
                    ap = average_precision_at_k(recommended_ids, user_future_clicks, k)
                    ndcg_val = dcg_at_k(recommended_ids, user_future_clicks, k)
                    ideal_dcg = dcg_at_k(list(user_future_clicks), user_future_clicks, min(k, len(user_future_clicks)))
                    ndcg = ndcg_val / ideal_dcg if ideal_dcg > 0 else 0.0

                    # Accumulate for cluster-level summary.
                    user_metrics[k]["precision"].append(prec)
                    user_metrics[k]["recall"].append(rec)
                    user_maps[k].append(ap)
                    user_ndcgs[k].append(ndcg)
                    
                    # Record per-user metrics along with number of recommendations.
                    partial_result_row[f"precision_{k}"] = prec
                    partial_result_row[f"recall_{k}"] = rec
                    partial_result_row[f"ap_{k}"] = ap
                    partial_result_row[f"ndcg_{k}"] = ndcg
                    partial_result_row[f"num_recommendations_{k}"] = len(recommended_ids)
                
                partial_result_row["status"] = "DONE"
                partial_buffer.append(partial_result_row)
                
                logging.error(f"iter:{i}")
                # Flush partial results every x users.
                if (i + 1) % 10 == 0:
                    logging.error(f"Writing partial rows: {partial_buffer}")
                    write_partial_rows(partial_buffer, partial_csv)
                    partial_buffer = []
                    
            except Exception as e:
                logging.error(f"Failed on user {user_id} in cluster {cluster_id} with error: {e}\n{format_exc()}")
                partial_buffer.append({
                    "cluster_id": cluster_id,
                    "user_id": user_id,
                    "user_index_in_cluster": i,
                    "status": f"FAILED: {e}"
                })
                continue

        # Flush any remaining rows for this cluster.
        if partial_buffer:
            write_partial_rows(partial_buffer, partial_csv)
            partial_buffer = []
        
        # Summarize cluster-level metrics.
        coverage_cluster = compute_coverage(cluster_recs, total_articles)
        for k in k_values:
            mean_precision = np.mean(user_metrics[k]["precision"]) if user_metrics[k]["precision"] else 0
            mean_recall = np.mean(user_metrics[k]["recall"]) if user_metrics[k]["recall"] else 0
            map_k = np.mean(user_maps[k]) if user_maps[k] else 0
            ndcg_k = np.mean(user_ndcgs[k]) if user_ndcgs[k] else 0

            results.append({
                "cluster_id": cluster_id,
                "k": k,
                "precision_user_level": mean_precision,
                "recall_user_level": mean_recall,
                "MAP": map_k,
                "nDCG": ndcg_k,
                "coverage": coverage_cluster, 
                "num_users": len(user_list)
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("user_level_experiment_results.csv", index=False)
    print("Cluster-level experiment results saved to 'user_level_experiment_results.csv'")
    return results_df

# --- [Data Preparation Function] ---
def prepare_train_df(
    data_dir,
    news_file,
    behaviours_file,
    user_category_profiles,
    num_clusters=3,
    fraction=1,
    max_title_length=30,
    max_history_length=50,
    downsampling=False
):
    # Load news data
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    print("Loaded news data:")
    print(news_df.head())

    # Load behaviors data
    behaviors_path = os.path.join(data_dir, behaviours_file)
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    print("Loaded behaviors data:")
    print(behaviors_df.head())

    # Clean titles and abstracts
    news_df['CleanTitle'] = news_df['Title'].apply(clean_text)
    news_df['CleanAbstract'] = news_df['Abstract'].apply(clean_text)

    # Create a combined text field
    news_df['CombinedText'] = news_df['CleanTitle'] + ' ' + news_df['CleanAbstract']
    news_df["CombinedText"] = news_df["CombinedText"].astype(str)
    news_df["CombinedText"] = news_df["CombinedText"].fillna("")


    # Initialize and fit tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")

    # Save tokenizer for future use
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Encode and pad CombinedText
    news_df['EncodedText'] = tokenizer.texts_to_sequences(news_df['CombinedText'])
    news_df['PaddedText'] = list(pad_sequences(news_df['EncodedText'], maxlen=max_title_length, padding='post', truncating='post'))

    # Create a mapping from NewsID to PaddedText
    news_text_dict = dict(zip(news_df['NewsID'], news_df['PaddedText']))

    # Function to parse impressions and labels
    def parse_impressions(impressions):
        impression_list = impressions.split()
        news_ids = []
        labels = []
        for imp in impression_list:
            try:
                news_id, label = imp.split('-')
                news_ids.append(news_id)
                labels.append(int(label))
            except ValueError:
                # Handle cases where split does not result in two items
                continue
        return news_ids, labels

    # Apply parsing to behaviors data
    behaviors_df[['ImpressionNewsIDs', 'ImpressionLabels']] = behaviors_df['Impressions'].apply(
        lambda x: pd.Series(parse_impressions(x))
    )

    # Initialize list for train samples
    train_samples = []

    # Iterate over behaviors to create train samples
    for _, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0]):
        user_id = row['UserID']
        user_cluster = row['Cluster'] if 'Cluster' in row else None  # Cluster will be assigned later

        # Parse user history
        history_ids = row['HistoryText'].split() if pd.notna(row['HistoryText']) else []
        history_texts = [news_text_dict.get(nid, [0]*max_title_length) for nid in history_ids]

        # Limit history length
        if len(history_texts) < max_history_length:
            padding = [[0]*max_title_length] * (max_history_length - len(history_texts))
            history_texts = padding + history_texts
        else:
            history_texts = history_texts[-max_history_length:]

        candidate_news_ids = row['ImpressionNewsIDs']
        labels = row['ImpressionLabels']

        for candidate_news_id, label in zip(candidate_news_ids, labels):
            candidate_text = news_text_dict.get(candidate_news_id, [0]*max_title_length)
            train_samples.append({
                'UserID': user_id,
                'HistoryTitles': history_texts,  # Renamed to 'HistoryTitles'
                'CandidateTitleTokens': candidate_text,  # Renamed to match DataGenerator
                'Label': label
            })

    # Create DataFrame from samples
    train_df = pd.DataFrame(train_samples)
    print(f"Created train_df with {len(train_df)} samples.")
    print("Columns in train_df:")
    print(train_df.columns)
    # --- [Clustering Users] ---
    # Ensure 'UserID's match between user_category_profiles and behaviors_df
    unique_user_ids = behaviors_df['UserID'].unique()
    user_category_profiles = user_category_profiles.loc[unique_user_ids]

    # Check for any missing 'UserID's
    missing_user_ids = set(unique_user_ids) - set(user_category_profiles.index)
    if missing_user_ids:
        print(f"Warning: {len(missing_user_ids)} 'UserID's are missing from user_category_profiles.")
        # Optionally handle missing users
        # For this example, we'll remove these users from behaviors_df
        behaviors_df = behaviors_df[~behaviors_df['UserID'].isin(missing_user_ids)]
    else:
        print("All 'UserID's are present in user_category_profiles.")

    # Perform clustering on user_category_profiles
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    user_clusters = kmeans.fit_predict(user_category_profiles)
    print(f"Assigned clusters to users. Number of clusters: {num_clusters}")

    # Create a DataFrame for user clusters
    user_cluster_df = pd.DataFrame({
        'UserID': user_category_profiles.index,
        'Cluster': user_clusters
    })

    # --- [Assign Clusters Using Map] ---
    print("Assigning cluster labels to train_df using map...")
    user_cluster_mapping = dict(zip(user_cluster_df['UserID'], user_cluster_df['Cluster']))
    train_df['Cluster'] = train_df['UserID'].map(user_cluster_mapping)

    # Verify cluster assignment
    missing_clusters = train_df[train_df['Cluster'].isna()]
    if not missing_clusters.empty:
        print(f"Warning: {len(missing_clusters)} samples have missing cluster assignments.")
        # Remove samples with missing cluster assignments
        train_df = train_df.dropna(subset=['Cluster'])
    else:
        print("All samples have cluster assignments.")

    # Convert 'Cluster' column to integer type
    train_df['Cluster'] = train_df['Cluster'].astype(int)
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} test samples.")
    # Assuming train_df has a 'Cluster' column indicating cluster assignments
    # Find the minimum size among all clusters
    min_cluster_size = train_df['Cluster'].value_counts().min()

    # Initialize an empty list to hold balanced data
    balanced_data = []

    # Iterate over each cluster and sample data to balance
    for cluster in train_df['Cluster'].unique():
        cluster_data = train_df[train_df['Cluster'] == cluster]
        balanced_cluster_data = cluster_data.sample(n=min_cluster_size, random_state=42)
        balanced_data.append(balanced_cluster_data)

    # Combine balanced data for all clusters
    balanced_train_df = pd.concat(balanced_data)
    print("\nlabel balance (0 vs 1):")
    print(train_df['Label'].value_counts())
    if downsampling:
        # Update train_df with the balanced data
        # --- [Label Balancing for 0/1 Classes] ---
        # Count how many 0s and 1s we have
        label_counts = balanced_train_df['Label'].value_counts()
        min_label_count = label_counts.min()

        balanced_labels = []
        for label_value in balanced_train_df['Label'].unique():
            label_data = balanced_train_df[balanced_train_df['Label'] == label_value]
            # Downsample to the min_label_count to balance the label distribution
            balanced_label_data = label_data.sample(n=min_label_count, random_state=42)
            balanced_labels.append(balanced_label_data)

        # Combine the label balanced data
        final_balanced_train_df = pd.concat(balanced_labels, ignore_index=True)

        # Shuffle the final dataset to mix up the rows
        final_balanced_train_df = final_balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print("\nAfter label balancing (0 vs 1):")
        print(final_balanced_train_df['Label'].value_counts())

        # Now final_balanced_train_df is balanced both by cluster and by label
        train_df = final_balanced_train_df

    #train_df = balanced_train_df.reset_index(drop=True)

    # Print summary of the balanced dataset
    print("Balanced cluster sizes:")
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} samples")
    print("Balanced dataset:")
    print(train_df['Cluster'].value_counts())
    """
    clustered_data_balanced = {}
    min_cluster_size = float('inf')
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} test samples.")
        min_cluster_size = len(cluster_data) if len(cluster_data) < min_cluster_size else min_cluster_size

    for cluster in range(num_clusters):
        data = train_df[train_df['Cluster'] == cluster]
        if len(data) > min_cluster_size:
            clustered_data_balanced[cluster] = data.sample(n=min_cluster_size, random_state=42)
        else:
            clustered_data_balanced[cluster] = data

    print("Balanced cluster sizes:")
    for cluster, data in clustered_data_balanced.items():
        print(f"Cluster {cluster}: {len(data)} samples")
    """
    # --- [Sampling] ---
    # Optionally perform random sampling
    print(f"Original size: {len(train_df)}")
    train_df_sampled = train_df.sample(frac=fraction, random_state=42)
    print(f"Sampled size: {len(train_df_sampled)}")

    # Optionally, set train_df to sampled
    train_df = train_df_sampled
    print("Columns in sampled train_df:")
    print(train_df.columns)
    print(f"Cluster:{train_df['Cluster']}")
    # --- [Split Data for Each Cluster] ---
    print("Splitting data into training and validation sets for each cluster...")
    clustered_data = {}
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]

        if cluster_data.empty:
            print(f"No data for Cluster {cluster}. Skipping...")
            continue  # Skip to the next cluster

        train_data, val_data = train_test_split(cluster_data, test_size=0.2, random_state=42)
        clustered_data[cluster] = {
            'train': train_data.reset_index(drop=True),
            'val': val_data.reset_index(drop=True)
        }
        print(f"Cluster {cluster}: {len(train_data)} training samples, {len(val_data)} validation samples.")
    if "small" in data_dir:
        news_df_pkl = "models/small_news_df_processed.pkl"
        train_df_pkl = "models/small_train_df_processed.pkl"
    else:
        news_df_pkl = "models/news_df_processed.pkl"
        train_df_pkl = "models/train_df_processed.pkl"
    print("Saved after processing: models/news_df_processed.pkl, models/train_df_processed.pkl")
    news_df.to_pickle(news_df_pkl)
    train_df.to_pickle(train_df_pkl)

    return clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters

# --- [DataGenerator Class] ---
class DataGenerator(Sequence):
    def __init__(self, df, batch_size, max_history_length=50, max_title_length=30):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.max_history_length = max_history_length
        self.max_title_length = max_title_length
        self.indices = np.arange(len(self.df))
        #print(f"[DataGenerator] Initialized with {len(self.df)} samples and batch_size={self.batch_size}")

    def __len__(self):
        length = int(np.ceil(len(self.df) / self.batch_size))
        #print(f"[DataGenerator] Number of batches per epoch: {length}")
        return length

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.df))
        batch_indices = self.indices[start:end]
        batch_df = self.df.iloc[batch_indices]

        # Debugging: Print batch information
        #print(f"[DataGenerator] Generating batch {idx+1}/{self.__len__()} with samples {start} to {end}")

        if len(batch_df) == 0:
            print(f"[DataGenerator] Warning: Batch {idx} is empty.")
            return None, None

        # Initialize batches
        history_batch = []
        candidate_batch = []
        labels_batch = []

        for _, row in batch_df.iterrows():
            # Get tokenized history titles
            history_titles = row['HistoryTitles']  # List of lists of integers

            # Pad each title in history
            history_titles_padded = pad_sequences(
                history_titles,
                maxlen=self.max_title_length,
                padding='post',
                truncating='post',
                value=0
            )

            # Pad or truncate the history to MAX_HISTORY_LENGTH
            if len(history_titles_padded) < self.max_history_length:
                padding = np.zeros((self.max_history_length - len(history_titles_padded), self.max_title_length), dtype='int32')
                history_titles_padded = np.vstack([padding, history_titles_padded])
            else:
                history_titles_padded = history_titles_padded[-self.max_history_length:]

            # Get candidate title tokens
            candidate_title = row['CandidateTitleTokens']  # List of integers
            candidate_title_padded = pad_sequences(
                [candidate_title],
                maxlen=self.max_title_length,
                padding='post',
                truncating='post',
                value=0
            )[0]

            # Append to batches
            history_batch.append(history_titles_padded)
            candidate_batch.append(candidate_title_padded)
            labels_batch.append(row['Label'])

        # Convert to numpy arrays
        history_batch = np.array(history_batch, dtype='int32')  # Shape: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH)
        candidate_batch = np.array(candidate_batch, dtype='int32')  # Shape: (batch_size, MAX_TITLE_LENGTH)
        labels_batch = np.array(labels_batch, dtype='float32')  # Shape: (batch_size,)
        inputs = {
            'history_input': history_batch,
            'candidate_input': candidate_batch
        }

        # Debugging: Print shapes
        #print(f"[DataGenerator] Batch shapes - history_input: {history_batch.shape}, candidate_input: {candidate_batch.shape}, labels: {labels_batch.shape}")
        return inputs, labels_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# --- [Fastformer Model Classes and Functions] ---
from tensorflow.keras.layers import Layer, Dense, Dropout, Softmax, Multiply, Embedding, TimeDistributed, LayerNormalization
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class SqueezeLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super(SqueezeLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

@register_keras_serializable()
class ExpandDimsLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({'axis': self.axis})
        return config

@register_keras_serializable()
class SumPooling(Layer):
    def __init__(self, axis=1, **kwargs):
        super(SumPooling, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        config = super(SumPooling, self).get_config()
        config.update({'axis': self.axis})
        return config

@register_keras_serializable()
class Fastformer(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        super(Fastformer, self).__init__(**kwargs)
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head

        self.WQ = None
        self.WK = None
        self.WV = None
        self.WO = None

    def build(self, input_shape):
        self.WQ = Dense(self.output_dim, use_bias=False, name='WQ')
        self.WK = Dense(self.output_dim, use_bias=False, name='WK')
        self.WV = Dense(self.output_dim, use_bias=False, name='WV')
        self.WO = Dense(self.output_dim, use_bias=False, name='WO')
        super(Fastformer, self).build(input_shape)

    def call(self, inputs):
        if len(inputs) == 2:
            Q_seq, K_seq = inputs
            Q_mask = None
            K_mask = None
        elif len(inputs) == 4:
            Q_seq, K_seq, Q_mask, K_mask = inputs

        batch_size = tf.shape(Q_seq)[0]
        seq_len = tf.shape(Q_seq)[1]

        # Linear projections
        Q = self.WQ(Q_seq)  # Shape: (batch_size, seq_len, output_dim)
        K = self.WK(K_seq)  # Shape: (batch_size, seq_len, output_dim)
        V = self.WV(K_seq)  # Shape: (batch_size, seq_len, output_dim)

        # Reshape for multi-head attention
        Q = tf.reshape(Q, (batch_size, seq_len, self.nb_head, self.size_per_head))
        K = tf.reshape(K, (batch_size, seq_len, self.nb_head, self.size_per_head))
        V = tf.reshape(V, (batch_size, seq_len, self.nb_head, self.size_per_head))

        # Compute global query and key
        global_q = tf.reduce_mean(Q, axis=1, keepdims=True)  # (batch_size, 1, nb_head, size_per_head)
        global_k = tf.reduce_mean(K, axis=1, keepdims=True)  # (batch_size, 1, nb_head, size_per_head)

        # Compute attention weights
        weights = global_q * K + global_k * Q  # (batch_size, seq_len, nb_head, size_per_head)
        weights = tf.reduce_sum(weights, axis=-1)  # (batch_size, seq_len, nb_head)
        weights = tf.nn.softmax(weights, axis=1)  # Softmax over seq_len

        # Apply attention weights to values
        weights = tf.expand_dims(weights, axis=-1)  # (batch_size, seq_len, nb_head, 1)
        context = weights * V  # (batch_size, seq_len, nb_head, size_per_head)

        # Combine heads
        context = tf.reshape(context, (batch_size, seq_len, self.output_dim))

        # Final projection
        output = self.WO(context)  # (batch_size, seq_len, output_dim)

        return output  # Output shape: (batch_size, seq_len, output_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super(Fastformer, self).get_config()
        config.update({
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head
        })
        return config

@register_keras_serializable()
class NewsEncoder(Layer):
    def __init__(self, vocab_size, embedding_dim=256, dropout_rate=0.2, nb_head=8, size_per_head=32, embedding_layer=None, **kwargs):
        super(NewsEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.nb_head = nb_head
        self.size_per_head = size_per_head

        # Define sub-layers
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name='embedding_layer'
        )
        self.dropout = Dropout(self.dropout_rate)
        self.dense = Dense(1)
        self.softmax = Softmax(axis=1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

        self.fastformer_layer = Fastformer(nb_head=self.nb_head, size_per_head=self.size_per_head, name='fastformer_layer')

    def build(self, input_shape):
        super(NewsEncoder, self).build(input_shape)

    def call(self, inputs):
        # Create mask
        mask = tf.cast(tf.not_equal(inputs, 0), dtype='float32')  # Shape: (batch_size, seq_len)

        # Embedding
        title_emb = self.embedding_layer(inputs)  # Shape: (batch_size, seq_len, embedding_dim)
        title_emb = self.dropout(title_emb)

        # Fastformer
        hidden_emb = self.fastformer_layer([title_emb, title_emb, mask, mask])  # Shape: (batch_size, seq_len, embedding_dim)
        hidden_emb = self.dropout(hidden_emb)

        # Attention-based Pooling
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, seq_len, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, seq_len)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, seq_len, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, seq_len, embedding_dim)
        news_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)

        return news_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(NewsEncoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head,
            'embedding_layer': tf.keras.utils.serialize_keras_object(self.embedding_layer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_layer_config = config.pop('embedding_layer', None)
        embedding_layer = tf.keras.layers.deserialize(embedding_layer_config) if embedding_layer_config else None
        return cls(embedding_layer=embedding_layer, **config)

@register_keras_serializable()
class MaskLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Create mask: cast to float32 any position that is not equal to zero
        mask = tf.cast(tf.not_equal(inputs, 0), dtype='float32')
        return mask

    def get_config(self):
        config = super(MaskLayer, self).get_config()
        return config

@register_keras_serializable()
class UserEncoder(Layer):
    def __init__(self, news_encoder_layer, embedding_dim=256, **kwargs):
        super(UserEncoder, self).__init__(**kwargs)
        self.news_encoder_layer = news_encoder_layer
        self.embedding_dim = embedding_dim
        self.dropout = Dropout(0.2)
        self.layer_norm = LayerNormalization()
        self.fastformer = Fastformer(nb_head=8, size_per_head=32, name='user_fastformer')
        self.dense = Dense(1)
        self.squeeze = SqueezeLayer(axis=-1)
        self.softmax = Softmax(axis=1)
        self.expand_dims = ExpandDimsLayer(axis=-1)
        self.sum_pooling = SumPooling(axis=1)

    def call(self, inputs):
        # inputs: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH)
        # Encode each news article in the history
        news_vectors = TimeDistributed(self.news_encoder_layer)(inputs)  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)

        # Step 1: Create a boolean mask
        mask = tf.not_equal(inputs, 0)  # Shape: (batch_size, MAX_HISTORY_LENGTH, MAX_TITLE_LENGTH), dtype=bool

        # Step 2: Reduce along the last axis
        mask = tf.reduce_any(mask, axis=-1)  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=bool

        # Step 3: Cast to float32 if needed
        mask = tf.cast(mask, dtype='float32')  # Shape: (batch_size, MAX_HISTORY_LENGTH), dtype=float32

        # Fastformer
        hidden_emb = self.fastformer([news_vectors, news_vectors, mask, mask])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        hidden_emb = self.dropout(hidden_emb)
        hidden_emb = self.layer_norm(hidden_emb)

        # Attention-based Pooling over history
        attention_scores = self.dense(hidden_emb)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        attention_scores = self.squeeze(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, MAX_HISTORY_LENGTH)
        attention_weights = self.expand_dims(attention_weights)  # Shape: (batch_size, MAX_HISTORY_LENGTH, 1)
        multiplied = Multiply()([hidden_emb, attention_weights])  # Shape: (batch_size, MAX_HISTORY_LENGTH, embedding_dim)
        user_vector = self.sum_pooling(multiplied)  # Shape: (batch_size, embedding_dim)

        return user_vector  # Shape: (batch_size, embedding_dim)

    def get_config(self):
        config = super(UserEncoder, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'news_encoder_layer': tf.keras.utils.serialize_keras_object(self.news_encoder_layer),
        })
        return config
    @classmethod
    def from_config(cls, config):
        # Extract the serialized news_encoder_layer config
        news_encoder_config = config.pop("news_encoder_layer")
        # Reconstruct the news_encoder_layer instance
        news_encoder_layer = tf.keras.utils.deserialize_keras_object(
            news_encoder_config, custom_objects={'NewsEncoder': NewsEncoder}
        )
        return cls(news_encoder_layer, **config)

def build_model(vocab_size, max_title_length=30, max_history_length=50, embedding_dim=256, nb_head=8, size_per_head=32, dropout_rate=0.2):
    # Define Inputs
    history_input = Input(shape=(max_history_length, max_title_length), dtype='int32', name='history_input')
    candidate_input = Input(shape=(max_title_length,), dtype='int32', name='candidate_input')

    # Instantiate NewsEncoder Layer
    news_encoder_layer = NewsEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        nb_head=nb_head,
        size_per_head=size_per_head,
        name='news_encoder'
    )

    # Encode Candidate News
    candidate_vector = news_encoder_layer(candidate_input)  # Shape: (batch_size, embedding_dim)

    # Encode User History
    user_vector = UserEncoder(news_encoder_layer, embedding_dim=embedding_dim, name='user_encoder')(history_input)  # Shape: (batch_size, embedding_dim)

    # Scoring Function: Dot Product between User and Candidate Vectors
    score = Dot(axes=-1)([user_vector, candidate_vector])  # Shape: (batch_size, 1)
    score = Activation('sigmoid')(score)  # Shape: (batch_size, 1)

    # Build Model
    model = Model(inputs={'history_input': history_input, 'candidate_input': candidate_input}, outputs=score)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='AUC')
        ]
    )

    return model

def train_cluster_models(clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters, batch_size=64, epochs=5, load_models=[]):
    models = {}

    for cluster in range(num_clusters):
        m_name = f'fastformer_cluster_{cluster}_full_balanced_1_epoch'
        weights_file = f'{m_name}.weights.h5'
        model_file = f'{m_name}.keras'
        model_h5_file = f'{m_name}.h5'
        model_hdf5_file = f'{m_name}.hdf5'
        model_json_file = f'{m_name}.json'
        if cluster in load_models: # load_models should be list of number indicating which models to load and not train
            print(f"\nLoading model for Cluster {cluster} from {model_file}")
            local_model_path = hf_hub_download(
                repo_id=f"Teemu5/news",
                filename=model_file,
                local_dir="."
            )
            import tensorflow as tf
            import keras
            import os
            entries = os.listdir('.')
            for entry in entries:
                print(entry)
            print(tf.__version__)
            print(keras.__version__)
            from tensorflow.keras.utils import custom_object_scope
            with custom_object_scope({'UserEncoder': UserEncoder, 'NewsEncoder': NewsEncoder}):
                model = tf.keras.models.load_model(model_file)#build_and_load_weights(weights_file)
                models[cluster] = model
            #model.save(model_file)
            #print(f"Saved model for Cluster {cluster} into {model_file}.")
            continue
        print(f"\nTraining model for Cluster {cluster} into {weights_file}")
        # Retrieve training and validation data
        train_data = clustered_data[cluster]['train']
        val_data = clustered_data[cluster]['val']

        print(f"Cluster {cluster} - Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

        # Create data generators
        train_generator = DataGenerator(train_data, batch_size, max_history_length, max_title_length)
        val_generator = DataGenerator(val_data, batch_size, max_history_length, max_title_length)

        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)

        # Build model
        model = build_model(
            vocab_size=vocab_size,
            max_title_length=max_title_length,
            max_history_length=max_history_length,
            embedding_dim=256,
            nb_head=8,
            size_per_head=32,
            dropout_rate=0.2
        )
        print(model.summary())

        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_AUC',
            patience=2,
            mode='max',
            restore_best_weights=True
        )
        csv_logger = CSVLogger(f'training_log_cluster_{cluster}.csv', append=True)
        model_checkpoint = ModelCheckpoint(
            f'best_model_cluster_{cluster}.keras',
            monitor='val_AUC',
            mode='max',
            save_best_only=True
        )
        print("\nlabel balance (0 vs 1):")
        print(train_df['Label'].value_counts())
        class_weight = get_class_weights(train_df['Label'])
        print("Class weights:", class_weight)
        logging.info(f"Class weights: {class_weight}")
        # Train the model
        model.fit(
            train_generator,
            epochs=epochs,
            #steps_per_epoch=steps_per_epoch,
            #validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[early_stopping, csv_logger, model_checkpoint],
            class_weight=class_weight
        )

        # Save model weights
        model.save_weights(weights_file)
        print(f"Saved model weights for Cluster {cluster} into {weights_file}.")
        model.save(model_h5_file)
        print(f"Saved h5 model for Cluster {cluster} into {model_h5_file}.")
        model.save(model_file)
        print(f"Saved model for Cluster {cluster} into {model_file}.")

        # Store the model
        models[cluster] = model
        # Clear memory
        del train_data, val_data, train_generator, val_generator, model
        import gc
        gc.collect()
    print("Returning models list")
    print(models)
    return models
def is_colab():
    return 'COLAB_GPU' in os.environ
def make_clustered_data(train_df, num_clusters):
    clustered_data = {}
    for cluster in range(num_clusters):
        cluster_data = train_df[train_df['Cluster'] == cluster]

        if cluster_data.empty:
            print(f"No data for Cluster {cluster}. Skipping...")
            continue  # Skip to the next cluster

        train_data, val_data = train_test_split(cluster_data, test_size=0.2, random_state=42)
        clustered_data[cluster] = {
            'train': train_data.reset_index(drop=True),
            'val': val_data.reset_index(drop=True)
        }
        print(f"Cluster {cluster}: {len(train_data)} training samples, {len(val_data)} validation samples.")
    return clustered_data
def make_user_cluster_df(user_category_profiles_path = 'user_category_profiles.pkl', user_cluster_df_path = 'user_cluster_df.pkl'):
    import pandas as pd
    import numpy as np
    import os
    from sklearn.cluster import KMeans
    import pickle

    # Load the user_category_profiles
    user_category_profiles = pd.read_pickle(user_category_profiles_path)

    # --- [Perform Clustering] ---

    # Optionally, standardize the features
    from sklearn.preprocessing import StandardScaler

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler on the user profiles
    user_profiles_scaled = scaler.fit_transform(user_category_profiles)

    # Save the scaler for future use
    scaler_path = 'user_profiles_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    # Initialize the clustering model
    num_clusters = 3  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Fit the clustering model
    kmeans.fit(user_profiles_scaled)

    # Save the clustering model for future use
    clustering_model_path = 'kmeans_user_clusters.pkl'
    with open(clustering_model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"Saved KMeans clustering model to {clustering_model_path}")

    # Assign clusters to users
    user_clusters = kmeans.predict(user_profiles_scaled)

    # Add the cluster assignments to the user profiles
    user_category_profiles['Cluster'] = user_clusters

    # Save the cluster assignments
    user_cluster_df = user_category_profiles[['Cluster']]
    user_cluster_df.to_pickle(user_cluster_df_path)
    print(f"Saved user cluster assignments to {user_cluster_df_path}")

# --- [Dataset Loading and Midpoint Calculation] ---
def load_dataset(data_dir, news_file='news.tsv', behaviors_file='behaviors.tsv'):
    """
    Loads news and behaviors files from the given directory.
    Fills missing HistoryText.
    Returns: news_df, behaviors_df
    """
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(news_path, sep='\t',
                          names=['NewsID','Category','SubCategory','Title','Abstract','URL','TitleEntities','AbstractEntities'],
                          index_col=False)
    print(f"Loaded news data from {news_path}: {news_df.shape}")

    behaviors_path = os.path.join(data_dir, behaviors_file)
    behaviors_df = pd.read_csv(behaviors_path, sep='\t',
                               names=['ImpressionID','UserID','Time','HistoryText','Impressions'],
                               index_col=False)
    print(f"Loaded behaviors data from {behaviors_path}: {behaviors_df.shape}")
    behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')
    return news_df, behaviors_df

def get_midpoint_time(behaviors_df):
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    min_time = behaviors_df['Time'].min()
    max_time = behaviors_df['Time'].max()
    midpoint = min_time + (max_time - min_time) / 2
    print(f"Midpoint time computed: {midpoint}")
    return midpoint

def init_dataset(data_dir, news_file='news.tsv', behaviors_file='behaviors.tsv'):
    """
    Loads a dataset from data_dir.
    Returns news_df and behaviors_df.
    """
    news_df, behaviors_df = load_dataset(data_dir, news_file, behaviors_file)
    # Create a combined text field for news
    news_df['CleanTitle'] = news_df['Title'].apply(clean_text)
    news_df['CleanAbstract'] = news_df['Abstract'].apply(clean_text)
    news_df['CombinedText'] = news_df['CleanTitle'] + ' ' + news_df['CleanAbstract']
    news_df["CombinedText"] = news_df["CombinedText"].astype(str).fillna("")
    return news_df, behaviors_df

# --- [Tokenizer and Preprocessing] ---
def prepare_tokenizer(news_df, max_title_length=30):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    # Save tokenizer for later use
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    return tokenizer, vocab_size

def init_bug(process_dfs=False, process_behaviors=False,
         data_dir='dataset/train/', valid_data_dir='dataset/valid/',
         zip_file="MINDlarge_train.zip", valid_zip_file="MINDlarge_dev.zip"):
    global vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, \
           user_category_profiles, clustered_data, tokenizer, num_clusters

    if is_colab():
        print("Running on Google Colab")
        data_dir = '/content/train/'
        valid_data_dir = '/content/valid/'

    # Construct full paths for the zip files
    zip_file_path = os.path.join(data_dir, zip_file)
    valid_zip_file_path = os.path.join(valid_data_dir, valid_zip_file)
    
    # Download zips if needed.
    if not os.path.exists(zip_file_path):
        hf_hub_download(repo_id="Teemu5/news", filename=zip_file, local_dir=data_dir)
    if not os.path.exists(valid_zip_file_path):
        hf_hub_download(repo_id="Teemu5/news", filename=valid_zip_file, local_dir=valid_data_dir)
    
    # Unzip both training and validation datasets.
    output_folder = os.path.dirname(zip_file_path)
    valid_output_folder = os.path.dirname(valid_zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    with zipfile.ZipFile(valid_zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(valid_output_folder)

    # Load datasets using our helper.
    news_df_train, behaviors_df_train = load_dataset(data_dir)
    news_df_valid, behaviors_df_valid = load_dataset(valid_data_dir)
    
    print("\n--- Training Data ---")
    print(news_df_train.head())
    print(behaviors_df_train.head())
    
    print("\n--- Validation Data ---")
    print(news_df_valid.head())
    print(behaviors_df_valid.head())
    
    # For backward compatibility, we continue processing only training data.
    # (You can later decide to combine or process validation data similarly.)
    news_df = news_df_train.copy()
    behaviors_df = behaviors_df_train.copy()
    
    # The rest of your original init() follows.
    # For example, initialize tokenizer on the training news:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    max_history_length = 50
    max_title_length = 30

    # If process_dfs is True, then call your prepare_train_df() etc.
    if process_dfs:
        (clustered_data, tokenizer, vocab_size, max_history_length,
         max_title_length, num_clusters) = prepare_train_df(
            data_dir=data_dir,
            news_file="news.tsv",
            behaviours_file="behaviors.tsv",
            user_category_profiles=None,  # Assuming you later generate or load profiles
            num_clusters=3,
            fraction=1,
            max_title_length=max_title_length,
            max_history_length=max_history_length
        )
    else:
        # Load preprocessed news and train DataFrames if available.
        news_df = pd.read_pickle("models/news_df_processed.pkl")
        train_df = pd.read_pickle("models/train_df_processed.pkl")
        clustered_data = make_clustered_data(train_df, num_clusters=3)
    
    # Return everything as before.
    return data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, \
           user_category_profiles, clustered_data, tokenizer, num_clusters

def unzip_datasets(data_dir, valid_data_dir, zip_file, valid_zip_file):
    zip_file_path = f"{data_dir}{zip_file}"
    valid_zip_file_path = f"{valid_data_dir}{valid_zip_file}"
    local_file_path = os.path.join(data_dir, zip_file)
    local_valid_file_path = os.path.join(valid_data_dir, valid_zip_file)
    if not os.path.exists(local_file_path):
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename=zip_file,
            local_dir=data_dir
        )
    if not os.path.exists(local_valid_file_path):
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename=valid_zip_file,
            local_dir=valid_data_dir
        )
    output_folder = os.path.dirname(zip_file_path)
    valid_output_folder = os.path.dirname(valid_zip_file_path)
    
    # Unzip the file
    print(f"unzip {local_file_path}")
    with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    print(f"unzip {local_valid_file_path}")
    with zipfile.ZipFile(local_valid_file_path, 'r') as zip_ref:
        zip_ref.extractall(valid_output_folder)
    if is_colab():
      valid_output_folder = os.path.dirname(valid_zip_file_path)
      with zipfile.ZipFile(valid_zip_file_path, 'r') as zip_ref:
          zip_ref.extractall(os.path.dirname(valid_output_folder))
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'

# REMOVE OLD IF NEW WORKS
def init(process_dfs = False, process_behaviors = False, data_dir = 'dataset/train/', valid_data_dir = 'dataset/valid/', zip_file = f"MINDlarge_train.zip", valid_zip_file = f"MINDlarge_dev.zip"):
    global vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters
    if is_colab():
        print("Running on Google colab")
        data_dir = '/content/train/'
        valid_data_dir = '/content/valid/'
    #data_dir = 'dataset/small/train/'  # Adjust path as necessary
    #zip_file = f"MINDsmall_train.zip"
    unzip_datasets(data_dir, valid_data_dir, zip_file, valid_zip_file)
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'
    
    # Load news data
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    
    print("Loaded news data:")
    print(news_df.head())
    
    # Load behaviors data
    behaviors_path = os.path.join(data_dir, behaviors_file)
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    
    print("\nLoaded behaviors data:")
    print(behaviors_df.head())

    valid_news_path = os.path.join(valid_data_dir, news_file)
    valid_news_df = pd.read_csv(
        valid_news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    
    print("Loaded news data:")
    print(valid_news_df.head())
    
    # Load behaviors data
    valid_behaviors_path = os.path.join(valid_data_dir, behaviors_file)
    valid_behaviors_df = pd.read_csv(
        valid_behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    
    print("\nLoaded behaviors data:")
    print(valid_behaviors_df.head())
    if process_behaviors:
        # Handle missing 'HistoryText' by replacing NaN with empty string
        behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')
        
        # Create a NewsID to Category mapping
        newsid_to_category = news_df.set_index('NewsID')['Category'].to_dict()
        
        # Function to extract categories from HistoryText
        def extract_categories(history_text):
            if not history_text:
                return []
            news_ids = history_text.split(' ')
            categories = [newsid_to_category.get(news_id, 'Unknown') for news_id in news_ids]
            return categories
        
        # Apply the function to extract categories
        behaviors_df['HistoryCategories'] = behaviors_df['HistoryText'].apply(extract_categories)
        
        print("\nSample HistoryCategories:")
        print(behaviors_df[['UserID', 'HistoryCategories']].head())
        from collections import defaultdict
        
        # Initialize a dictionary to hold category counts per user
        user_category_counts = defaultdict(lambda: defaultdict(int))
        
        # Populate the dictionary
        for idx, row in behaviors_df.iterrows():
            user_id = row['UserID']
            categories = row['HistoryCategories']
            for category in categories:
                user_category_counts[user_id][category] += 1
        
        # Convert the dictionary to a DataFrame
        user_category_profiles = pd.DataFrame(user_category_counts).T.fillna(0)
        
        # Optionally, rename columns to indicate category
        user_category_profiles.columns = [f'Category_{cat}' for cat in user_category_profiles.columns]
        
        print("\nCreated user_category_profiles:")
        print(user_category_profiles.head())
        print(f"\nShape of user_category_profiles: {user_category_profiles.shape}")
        # Handle missing 'HistoryText' by replacing NaN with empty string
        behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna('')
        
        # Create a NewsID to Category mapping
        newsid_to_category = news_df.set_index('NewsID')['Category'].to_dict()
        
        # Get all unique UserIDs from behaviors_df
        unique_user_ids = behaviors_df['UserID'].unique()
        
        # Function to extract categories from HistoryText
        def extract_categories(history_text):
            if not history_text:
                return []
            news_ids = history_text.split(' ')
            categories = [newsid_to_category.get(news_id, 'Unknown') for news_id in news_ids]
            return categories
        
        # Apply the function to extract categories
        behaviors_df['HistoryCategories'] = behaviors_df['HistoryText'].apply(extract_categories)
        
        # Explode 'HistoryCategories' to have one category per row
        behaviors_exploded = behaviors_df[['UserID', 'HistoryCategories']].explode('HistoryCategories')
        
        # Replace missing categories with 'Unknown'
        behaviors_exploded['HistoryCategories'] = behaviors_exploded['HistoryCategories'].fillna('Unknown')
        
        # Create a cross-tabulation (pivot table) of counts
        user_category_counts = pd.crosstab(
            index=behaviors_exploded['UserID'],
            columns=behaviors_exploded['HistoryCategories']
        )
        
        # Rename columns to include 'Category_' prefix
        user_category_counts.columns = [f'Category_{col}' for col in user_category_counts.columns]
        
        # Reindex to include all users, filling missing values with zero
        user_category_profiles = user_category_counts.reindex(unique_user_ids, fill_value=0)
        
        print(f"\nCreated user_category_profiles with {user_category_profiles.shape[0]} users and {user_category_profiles.shape[1]} categories.")
        
        # Determine top N categories
        top_n = 20
        category_counts = news_df['Category'].value_counts()
        top_categories = category_counts.nlargest(top_n).index.tolist()
        
        # Get the category names without the 'Category_' prefix
        user_category_columns = user_category_profiles.columns.str.replace('Category_', '')
        
        # Filter columns in user_category_profiles that are in top_categories
        filtered_columns = user_category_profiles.columns[user_category_columns.isin(top_categories)]
        
        # Create filtered_user_category_profiles with these columns
        filtered_user_category_profiles = user_category_profiles[filtered_columns]
        
        # Identify columns that are not in top_categories to sum them into 'Category_Other'
        other_columns = user_category_profiles.columns[~user_category_columns.isin(top_categories)]
        
        # Sum the 'Other' categories
        filtered_user_category_profiles['Category_Other'] = user_category_profiles[other_columns].sum(axis=1)
        
        # Now, get the actual categories present after filtering
        actual_categories = filtered_columns.str.replace('Category_', '').tolist()
        
        # Add 'Other' to the list
        actual_categories.append('Other')
        print(f"Number of new column names: {len(actual_categories)}")
        # Assign new column names
        filtered_user_category_profiles.columns = [f'Category_{cat}' for cat in actual_categories]
        print("\nFiltered user_category_profiles with Top N Categories and 'Other':")
        print(filtered_user_category_profiles.head())
        print(f"\nShape of filtered_user_category_profiles: {filtered_user_category_profiles.shape}")
        
        # Save the user_category_profiles to a file for future use
        if "small" in data_dir:
            user_category_profiles_path = 'small_user_category_profiles.pkl'
            behaviors_df_processed_path = "small_behaviors_df_processed.pkl"
        else:
            user_category_profiles_path = 'user_category_profiles.pkl'
            behaviors_df_processed_path = "behaviors_df_processed.pkl"
        
        filtered_user_category_profiles.to_pickle(user_category_profiles_path)
        user_category_profiles = filtered_user_category_profiles
        print(f"\nSaved user_category_profiles to {user_category_profiles_path}")
        behaviors_df.to_pickle(behaviors_df_processed_path)
        print(f"\nSaved behaviors_df to {behaviors_df_processed_path}")
    else:
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename="user_category_profiles.pkl",
            local_dir="models"
        )
        local_model_path = hf_hub_download(
            repo_id=f"Teemu5/news",
            filename="behaviors_df_processed.pkl",
            local_dir="models"
        )
        user_category_profiles = pd.read_pickle("models/user_category_profiles.pkl")
        behaviors_df = pd.read_pickle("models/behaviors_df_processed.pkl")
    print(f"Number of columns in user_category_profiles: {len(user_category_profiles.columns)}")
    # Number of unique users in behaviors_df
    unique_user_ids = behaviors_df['UserID'].unique()
    print(f"Number of unique users in behaviors_df: {len(unique_user_ids)}")
    # Number of unique users in behaviors_df
    unique_user_ids = behaviors_df['UserID'].unique()
    print(f"Number of unique users in behaviors_df: {len(unique_user_ids)}")
    
    # Number of users in user_category_profiles
    user_profile_ids = user_category_profiles.index.unique()
    print(f"Number of users in user_category_profiles: {len(user_profile_ids)}")
    
    # Find missing UserIDs
    missing_user_ids = set(unique_user_ids) - set(user_profile_ids)
    print(f"Number of missing UserIDs in user_category_profiles: {len(missing_user_ids)}")
    tokenizer = Tokenizer()
    num_clusters = 3
    if process_dfs:
        clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
            data_dir=data_dir,
            news_file=news_file,
            behaviours_file=behaviors_file,
            user_category_profiles=user_category_profiles,
            num_clusters=num_clusters,
            fraction=1,
            max_title_length=30,
            max_history_length=50
        )
    local_model_path = hf_hub_download(
        repo_id=f"Teemu5/news",
        filename="news_df_processed.pkl",
        local_dir="models"
    )
    local_model_path = hf_hub_download(
        repo_id=f"Teemu5/news",
        filename="train_df_processed.pkl",
        local_dir="models"
    )
    if "small" in data_dir:
        news_df_pkl = "models/small_news_df_processed.pkl"
        train_df_pkl = "models/small_train_df_processed.pkl"
    else:
        news_df_pkl = "models/news_df_processed.pkl"
        train_df_pkl = "models/train_df_processed.pkl"
    news_df = pd.read_pickle(news_df_pkl)
    train_df = pd.read_pickle(train_df_pkl)

    clustered_data = make_clustered_data(train_df, num_clusters)

    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    max_history_length = 50
    max_title_length = 30
    batch_size = 64 # Adjust as needed
    return data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(labels):
    print(f"get_class_weights: labels={labels}")
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

def get_models(process_dfs = False, process_behaviors = False, data_dir = 'dataset/train/', valid_data_dir = 'dataset/valid/', zip_file = f"MINDlarge_train.zip", valid_zip_file = f"MINDlarge_dev.zip"):
    news_file = 'news.tsv'
    behaviors_file = 'behaviors.tsv'
    data_dir, vocab_size, max_history_length, max_title_length, news_df, train_df, behaviors_df, user_category_profiles, clustered_data, tokenizer, num_clusters = init(process_dfs, process_behaviors, data_dir, valid_data_dir, zip_file, valid_zip_file)
    if process_dfs:
        clustered_data, tokenizer, vocab_size, max_history_length, max_title_length, num_clusters = prepare_train_df(
            data_dir=data_dir,
            news_file=news_file,
            behaviours_file=behaviors_file,
            user_category_profiles=user_category_profiles,
            num_clusters=3,
            fraction=1,
            max_title_length=30,
            max_history_length=50
        )
    models = train_cluster_models(
        clustered_data=clustered_data,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        max_history_length=max_history_length,
        max_title_length=max_title_length,
        num_clusters=num_clusters,
        batch_size=64,
        epochs=1,
        load_models=[0,1,2,3]
    )
    return models, news_df, behaviors_df, tokenizer

def train_test_split_time(behaviors_df, cutoff_str="2019-11-20"):
    # Ensure Time column is datetime64[ns] and timezone-naive
    if behaviors_df["Time"].dtype != "datetime64[ns]":
        behaviors_df["Time"] = pd.to_datetime(behaviors_df["Time"], errors="coerce")
    if behaviors_df["Time"].dt.tz is not None:
        behaviors_df["Time"] = behaviors_df["Time"].dt.tz_localize(None)

    # Also ensure cutoff_dt is timezone-naive
    cutoff_dt = pd.to_datetime(cutoff_str)
    if cutoff_dt.tzinfo is not None:
        cutoff_dt = cutoff_dt.tz_localize(None)

    train_data = behaviors_df[behaviors_df["Time"] <= cutoff_dt].copy()
    test_data = behaviors_df[behaviors_df["Time"] > cutoff_dt].copy()
    return train_data, test_data

# ===================== __main__ =====================
def main(dataset='train', process_dfs=False, process_behaviors=False,
         data_dir_train='dataset/train/', data_dir_valid='dataset/valid/',
         zip_file_train="MINDlarge_train.zip", zip_file_valid="MINDlarge_dev.zip",
         user_category_profiles_path='', user_cluster_df_path='', cluster_id=None):
    """
    Main function to run tests on a given dataset type ('train' or 'valid').
    It uses the midpoint time as cutoff and then runs evaluations.
    """
    # Choose dataset directory based on parameter.
    if dataset.lower() == 'train':
        data_dir = data_dir_train
    elif dataset.lower() == 'valid':
        data_dir = data_dir_valid
    else:
        raise ValueError("dataset must be either 'train' or 'valid'")
    
    # Unzip and load dataset (if not already unzipped)
    # (You can implement similar zip download/unzip logic if needed.)
    unzip_datasets(data_dir_train, data_dir_valid, zip_file_train, zip_file_valid)
    news_df, behaviors_df = init_dataset(data_dir)
    
    # Prepare tokenizer (we train only on news from the chosen dataset)
    tokenizer, vocab_size = prepare_tokenizer(news_df)
    max_history_length = 50
    max_title_length = 30

    # Compute the midpoint time from the behaviors data and use it as cutoff
    midpoint_time = get_midpoint_time(behaviors_df)
    # Format the time to ISO 8601 with a trailing 'Z' (assuming UTC)
    cutoff_time_str = midpoint_time.isoformat().replace('+00:00', 'Z')
    print("Using cutoff time:", cutoff_time_str)

    # (Assuming TF-IDF vectorizer is built on the news text)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(news_df["CombinedText"])
    
    # Here you would continue with any clustering steps or training steps.
    # For demonstration, we call the train_test_split function to get train/test splits from behaviors_df.
    train_data, test_data = train_test_split_time(behaviors_df, cutoff_time_str)

    # Here we assume your models_dict is obtained from training or loaded
    models_dict, news_df, behaviors_df, tokenizer = get_models(process_dfs, process_behaviors, data_dir_train, data_dir_valid, zip_file_train, zip_file_valid)
    # (Assume cluster_mapping and models_dict are available from your training procedure.
    #  For example, if you already saved your models and cluster assignments, you would load them here.)
    # For this example, we assume you have a user_cluster_df file:
    if user_cluster_df_path == '':
        user_cluster_df_path = hf_hub_download(
            repo_id="Teemu5/news",
            filename="user_cluster_df.pkl",
            local_dir="models"
        )
    else:
        # Optionally generate it if not available.
        make_user_cluster_df(user_category_profiles_path, user_cluster_df_path)
    user_cluster_df = pd.read_pickle(user_cluster_df_path)
    cluster_mapping = {}
    for cluster in user_cluster_df['Cluster'].unique():
        cluster_mapping[cluster] = user_cluster_df[user_cluster_df['Cluster'] == cluster].index.tolist()
    clusters_to_run = [999]
    if cluster_id is not None:
        # Support comma-separated list of cluster IDs:
        clusters_to_run = [int(x.strip()) for x in str(cluster_id).split(",")]
        cluster_mapping = {cl: users for cl, users in cluster_mapping.items() if cl in clusters_to_run}
        print("Processing only clusters:", list(cluster_mapping.keys()))
    # --- Run Evaluation and Write Intermediate Results ---
    # The user-level evaluation function writes intermediate partial results every 10 users.
    results_partial_csv = f"user_level_partial_results_{date_str}_{list(cluster_mapping.keys())[0]}.csv"
    results_user_level = run_cluster_experiments_user_level(
        cluster_mapping, 
        train_data, 
        test_data,
        news_df,
        behaviors_df,
        models_dict,
        tokenizer,
        tfidf_vectorizer,
        cutoff_time_str,
        k_values=[5, 10, 20, 50, 100],
        partial_csv=results_partial_csv,
        shuffle_clusters=False
    )

    cluster_results_df = run_cluster_experiments(
        cluster_mapping, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time_str, k_values=[5,10,20,50,100]
    )
    
    print("Evaluation complete. Intermediate results were written during testing.")
    return results_user_level, cluster_results_df

def main2(process_dfs = False, process_behaviors = False, data_dir = 'dataset/train/', valid_data_dir = 'dataset/valid/', zip_file = f"MINDlarge_train.zip", valid_zip_file = f"MINDlarge_dev.zip", user_category_profiles_path = '', user_cluster_df_path = ''):
    logging.info(f"Starting main(process_dfs = {process_dfs}, process_behaviors = {process_behaviors}, data_dir = {data_dir}, valid_data_dir = {valid_data_dir}, zip_file = {zip_file}, valid_zip_file = {valid_zip_file}, user_category_profiles_path = {user_category_profiles_path}, user_cluster_df_path = {user_cluster_df_path}):")
    # Assume get_models() returns models_dict, news_df, behaviors_df, tokenizer
    models_dict, news_df, behaviors_df, tokenizer = get_models(process_dfs, process_behaviors, data_dir, valid_data_dir, zip_file, valid_zip_file)
    logging.info(f"News data shape: {news_df.shape}")
    logging.info(f"Behaviors data shape: {behaviors_df.shape}")
    # Fit TF-IDF vectorizer on the combined text of news articles.
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(news_df["CombinedText"])
    
    # Assume you have a cluster mapping available.
    # For example, if you have a DataFrame 'user_cluster_df' with columns ['UserID', 'Cluster'],
    # then build a dictionary mapping cluster id -> list of user IDs:
    # Download the file from Hugging Face Hub if it's not available locally.
    if user_cluster_df_path == '':
        user_cluster_df_path = hf_hub_download(
            repo_id="Teemu5/news",
            filename="user_cluster_df.pkl",
            local_dir="models"
        )
    else:
        make_user_cluster_df(user_category_profiles_path, user_cluster_df_path)
    user_cluster_df = pd.read_pickle(user_cluster_df_path)
    cluster_mapping = {}
    for cluster in user_cluster_df['Cluster'].unique():
        cluster_mapping[cluster] = user_cluster_df[user_cluster_df['Cluster'] == cluster].index.tolist()
    
    # Set a cutoff time (simulate recommendations at this time)
    midpoint_time = get_midpoint_time(behaviors_df)
    cutoff_time_str = "2019-11-10T00:00:00Z"
    cutoff_time_str = midpoint_time.isoformat().replace('+00:00', 'Z')
    print("Earliest interaction:", behaviors_df['Time'].min())
    print("Latest interaction:",   behaviors_df['Time'].max())

    k_values = [5,10,20,50,100]
    train_data, test_data = train_test_split_time(behaviors_df, cutoff_time_str)
    results_user_level = run_cluster_experiments_user_level(
        cluster_mapping, 
        train_data, 
        test_data,
        news_df,
        behaviors_df,
        models_dict,
        tokenizer,
        tfidf_vectorizer,
        cutoff_time_str
    )
    # Run the cluster-level experiment
    cluster_results_df = run_cluster_experiments(cluster_mapping, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time_str, k_values)
