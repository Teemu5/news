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
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
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


def prepare_category_train_dfs(data_dir, news_file, behaviors_file, max_title_length=30, max_history_length=50, save_filename='category_train_dfs.pkl'):
    if os.path.exists(save_filename):
        print(f"Loading precomputed category training data from {save_filename}...")
        with open(save_filename, "rb") as f:
            category_train_dfs, news_df, behaviors_df, tokenizer = pickle.load(f)
        print("Loaded category training data.")
        return category_train_dfs, news_df, behaviors_df, tokenizer

    print("Precomputed category training data not found. Computing now...")

    # Load news data
    news_path = os.path.join(data_dir, news_file)
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'],
        index_col=False
    )
    print(f"Loaded news data from {news_path}: {news_df.shape}")

    # Clean titles and abstracts and create combined text
    news_df['CleanTitle'] = news_df['Title'].apply(clean_text)
    news_df['CleanAbstract'] = news_df['Abstract'].apply(clean_text)
    news_df['CombinedText'] = news_df['CleanTitle'] + " " + news_df['CleanAbstract']
    news_df["CombinedText"] = news_df["CombinedText"].astype(str).fillna("")

    # Load behaviors data
    behaviors_path = os.path.join(data_dir, behaviors_file)
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        names=['ImpressionID', 'UserID', 'Time', 'HistoryText', 'Impressions'],
        index_col=False
    )
    behaviors_df['HistoryText'] = behaviors_df['HistoryText'].fillna("")
    print(f"Loaded behaviors data from {behaviors_path}: {behaviors_df.shape}")

    # Fit the tokenizer on news CombinedText
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_df['CombinedText'].tolist())

    # Create mapping from NewsID to padded text
    news_df['EncodedText'] = tokenizer.texts_to_sequences(news_df['CombinedText'])
    news_df['PaddedText'] = list(pad_sequences(news_df['EncodedText'], maxlen=max_title_length, padding='post', truncating='post'))
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
                continue
        return news_ids, labels

    behaviors_df[['ImpressionNewsIDs', 'ImpressionLabels']] = behaviors_df['Impressions'].apply(
        lambda x: pd.Series(parse_impressions(x))
    )

    # Build training samples with candidate category information.
    samples = []
    print("Building training samples...")
    total_rows = len(behaviors_df)
    for i, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0], desc="Building samples"):
        user_id = row['UserID']
        # Process user history: get candidate training sample history
        history_ids = row['HistoryText'].split() if row['HistoryText'] != "" else []
        history_texts = [news_text_dict.get(nid, [0] * max_title_length) for nid in history_ids]
        if len(history_texts) < max_history_length:
            padding = [[0] * max_title_length] * (max_history_length - len(history_texts))
            history_texts = padding + history_texts
        else:
            history_texts = history_texts[-max_history_length:]

        candidate_news_ids = row['ImpressionNewsIDs']
        labels = row['ImpressionLabels']
        for candidate_news_id, label in zip(candidate_news_ids, labels):
            candidate_text = news_text_dict.get(candidate_news_id, [0] * max_title_length)
            # Look up candidate article category (if not found, use "Unknown")
            candidate_category_series = news_df[news_df['NewsID'] == candidate_news_id]['Category']
            candidate_category = candidate_category_series.iloc[0] if not candidate_category_series.empty else "Unknown"
            samples.append({
                'UserID': user_id,
                'HistoryTitles': history_texts,
                'CandidateTitleTokens': candidate_text,
                'Label': label,
                'CandidateCategory': candidate_category
            })
        if i % 1000 == 0:
            logging.info(f"{i+1}/{total_rows} rows done.")
    train_df = pd.DataFrame(samples)
    print(f"Created training DataFrame with {len(train_df)} samples.")
    
    # Group training samples by candidate category
    category_train_dfs = dict(tuple(train_df.groupby('CandidateCategory')))
    for category, df in category_train_dfs.items():
        print(f"Category '{category}': {len(df)} samples.")

    # Save the computed data to disk
    with open(save_filename, "wb") as f:
        pickle.dump((category_train_dfs, news_df, behaviors_df, tokenizer), f)
    print(f"Saved category training data to {save_filename}")

    return category_train_dfs, news_df, behaviors_df, tokenizer

def train_category_models(category_train_dfs, vocab_size, max_history_length, max_title_length, batch_size=64, epochs=5, dataset_size=''):
    """
    Train a model for each category in the category_train_dfs dict.
    """
    category_models = {}
    for category, df in category_train_dfs.items():
        print(f"--- Training model for category: {category} ---")
        # Split into train/validation sets
        train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
        print(f"Category '{category}': {len(train_data)} training samples, {len(val_data)} validation samples.")

        # Create data generators (assuming your DataGenerator class uses fields 'HistoryTitles' and 'CandidateTitleTokens')
        train_generator = DataGenerator(train_data, batch_size=batch_size, max_history_length=max_history_length, max_title_length=max_title_length)
        val_generator = DataGenerator(val_data, batch_size=batch_size, max_history_length=max_history_length, max_title_length=max_title_length)

        model = build_model(vocab_size, max_title_length, max_history_length, embedding_dim=256, nb_head=8, size_per_head=32, dropout_rate=0.2)
        model.summary(print_fn=lambda x: print(f"[{category}] {x}"))

        early_stopping = EarlyStopping(monitor='val_AUC', patience=2, mode='max', restore_best_weights=True)
        csv_logger = CSVLogger(f'training_log_category_{category}.csv', append=True)
        model_checkpoint = ModelCheckpoint(f'best_model_category_{category}.keras', monitor='val_AUC', mode='max', save_best_only=True)

        print(f"Training model for category '{category}'...")
        model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[early_stopping, csv_logger, model_checkpoint]
        )
        # Save model
        model_save_path = f'fastformer_{dataset_size}_category_{category}.keras'
        model.save(model_save_path)
        print(f"Saved model for category '{category}' to {model_save_path}")
        category_models[category] = model
    return category_models

def run_category_based_training(dataset_size, data_dir_train, valid_data_dir, zip_file, valid_zip_file, news_file='news.tsv', behaviors_file='behaviors.tsv'):
    print(f"unzipping datasets: data_dir_train={data_dir_train}, valid_data_dir={valid_data_dir}, zip_file={zip_file}, valid_zip_file={valid_zip_file}")
    unzip_datasets(data_dir_train, valid_data_dir, zip_file, valid_zip_file)
    print("Starting category-based training...")
    # Step 1: Load data and prepare training samples grouped by candidate category.
    category_train_dfs, news_df, behaviors_df, tokenizer = prepare_category_train_dfs(data_dir_train, news_file, behaviors_file, 30, 50, f"category_train_dfs_{dataset_size}.pkl")
    
    # Compute vocabulary size from the tokenizer.
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    max_history_length = 50
    max_title_length = 30
    
    # Step 2: Train a model for each category.
    category_models = train_category_models(category_train_dfs, vocab_size, max_history_length, max_title_length, batch_size=64, epochs=5, dataset_size=dataset_size)
    
    print("Category-based training complete.")
    return category_models, news_df, behaviors_df, tokenizer

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

def compute_precision_recall_at_k(recommended_ids, ground_truth_ids, k, candidate_ids=None):
    """
    Compute precision@k and recall@k after filtering ground truth to only include articles
    that exist in the candidate pool. If candidate_ids is not provided, it defaults to
    the union of recommended_ids and ground_truth_ids.
    """
    if candidate_ids is None:
        candidate_ids = list(set(recommended_ids).union(ground_truth_ids))
    filtered_ground_truth = set(ground_truth_ids).intersection(set(candidate_ids))
    recommended_k = recommended_ids[:k]
    if not filtered_ground_truth:
        return 0.0, 0.0
    relevant = [1 if rec in filtered_ground_truth else 0 for rec in recommended_k]
    precision = sum(relevant) / k
    recall = sum(relevant) / len(filtered_ground_truth)
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

def tfidf_filter_candidates(candidates_df: pd.DataFrame, user_history_text: str, tfidf_vectorizer: TfidfVectorizer, min_similarity: float = 0.1, plot = False) -> pd.DataFrame:
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
    if plot:
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
    k = min(k, len(candidate_ids))  # ensure k doesn't exceed available candidates
    top_indices = np.argsort(candidate_scores)[-k:][::-1]
    logging.info(f"k:{k} candidate_ids: {len(candidate_ids)}!!!")
    for i in top_indices:
        logging.info(f"i:{i} candidate_ids[i]: {candidate_ids[i]}")
    recommended_ids = [candidate_ids[i] for i in top_indices]
    return recommended_ids
def compute_precision_recall_at_k(recommended_ids, ground_truth_ids, k, candidate_ids=None):
    """
    Compute precision@k and recall@k after filtering ground truth to only include articles
    that exist in the candidate pool. If candidate_ids is not provided, it defaults to
    the union of recommended_ids and ground_truth_ids.
    """
    if candidate_ids is None:
        candidate_ids = list(set(recommended_ids).union(ground_truth_ids))
    filtered_ground_truth = set(ground_truth_ids).intersection(set(candidate_ids))
    recommended_k = recommended_ids[:k]
    if not filtered_ground_truth:
        return 0.0, 0.0
    relevant = [1 if rec in filtered_ground_truth else 0 for rec in recommended_k]
    precision = sum(relevant) / k
    recall = sum(relevant) / len(filtered_ground_truth)
    return precision, recall

def cluster_evaluate(recommended_ids, cluster_ground_truth, k):
    """
    Computes precision@k and recall@k for the cluster.
    """
    # Use the same compute_precision_recall_at_k function
    precision, recall = compute_precision_recall_at_k(recommended_ids, cluster_ground_truth, k)
    return precision, recall
def average_precision_at_k(recommended_ids, ground_truth_ids, k, candidate_ids=None):
    """
    Compute average precision at k after filtering ground truth to only include articles
    that exist in the candidate pool.
    """
    if candidate_ids is None:
        candidate_ids = list(set(recommended_ids).union(ground_truth_ids))
    filtered_ground_truth = set(ground_truth_ids).intersection(set(candidate_ids))
    recommended_k = recommended_ids[:k]
    hit_count = 0
    sum_precisions = 0.0
    for i, rec in enumerate(recommended_k, start=1):
        if rec in filtered_ground_truth:
            hit_count += 1
            sum_precisions += hit_count / i
    return sum_precisions / hit_count if hit_count > 0 else 0.0

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

def dcg_at_k(recommended_ids, ground_truth_ids, k, candidate_ids=None):
    """
    Compute Discounted Cumulative Gain at k after filtering ground truth.
    """
    if candidate_ids is None:
        candidate_ids = list(set(recommended_ids).union(ground_truth_ids))
    filtered_ground_truth = set(ground_truth_ids).intersection(set(candidate_ids))
    recommended_k = recommended_ids[:k]
    dcg = 0.0
    for i, rec in enumerate(recommended_k, start=1):
        if rec in filtered_ground_truth:
            dcg += 1.0 / math.log2(i + 1)
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

def get_user_shown_articles(user_id, behaviors_df, cutoff_time):
    """
    Return a set of article IDs that were shown to the user after cutoff_time,
    regardless of whether they were clicked.
    """
    shown_articles = set()
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff_time)
    future_rows = behaviors_df[
        (behaviors_df['UserID'] == user_id) &
        (behaviors_df['Time'] > np.datetime64(cutoff_dt))
    ]
    for _, row in future_rows.iterrows():
        if pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
            continue
        for imp in row["Impressions"].split():
            try:
                art_id, _ = imp.split('-')
                shown_articles.add(art_id)
            except:
                pass
    return shown_articles

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

def get_or_compute_global_average_profile(behaviors_df, news_df, tokenizer, cutoff_time, 
                                            max_history_length=50, max_title_length=30,
                                            filename='global_average_profile.pkl'):
    """
    Check if the global average profile has been computed and saved.
    If the file exists, load the global average profile.
    Otherwise, compute it, save it, and return it.
    
    Parameters:
      - behaviors_df: DataFrame of user behaviors.
      - news_df: DataFrame of news/articles.
      - tokenizer: Pre-fitted Keras Tokenizer.
      - cutoff_time: The cutoff time to split historical interactions.
      - max_history_length: Maximum number of historical items in the profile.
      - max_title_length: Maximum length of each article's tokenized title.
      - filename: The file in which the global average profile is stored.
      
    Returns:
      - A TensorFlow tensor of shape (max_history_length, max_title_length) representing the global average profile.
    """
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            avg_profile = pickle.load(f)
        print(f"Loaded global average profile from {filename}")
        return avg_profile
    else:
        # Compute the global average profile.
        profiles = []
        user_ids = behaviors_df['UserID'].unique()
        for user_id in user_ids:
            history_tensor, history_ids, _ = build_user_profile_tensor(user_id, behaviors_df, news_df, cutoff_time, tokenizer,
                                                                       max_history_length, max_title_length)
            if history_ids:  # Include only users with a non-empty history.
                profiles.append(history_tensor.numpy())
        if profiles:
            avg_profile = tf.convert_to_tensor(np.mean(np.stack(profiles, axis=0), axis=0), dtype=tf.int32)
        else:
            avg_profile = tf.zeros((max_history_length, max_title_length), dtype=tf.int32)
        # Save the computed global average profile.
        with open(filename, "wb") as f:
            pickle.dump(avg_profile, f)
        print(f"Saved global average profile to {filename}")
        return avg_profile

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
    logging.info(f"Started build_user_profile_tensor")
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff_time)
    logging.info(f"Time set to datetime")

    # Filter to user’s rows up to cutoff
    user_hist = behaviors_df[
        (behaviors_df['UserID'] == user_id)
        & (behaviors_df['Time'] <= np.datetime64(cutoff_dt))
    ].copy()
    logging.info(f"Filtered user behaviors up to {cutoff_time}")

    # Collect article IDs from user’s HistoryText
    all_hist_article_ids = set()
    for _, row in user_hist.iterrows():
        if pd.isna(row["HistoryText"]) or row["HistoryText"].strip() == "":
            continue
        for art in row["HistoryText"].split():
            all_hist_article_ids.add(art)
    logging.info(f"Collected users history of length:{len(all_hist_article_ids)}")

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
    logging.info(f"Kept users history of length:{len(history_article_ids)}")

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
    logging.info(f"Built combined text from every article in history. history_titles length:{len(history_titles)}")
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

    logging.info(f"Convert history_titles to tensor")
    # The final shape should be (max_history_length, max_title_length).
    # Convert to a TF tensor if you want:
    history_tensor = tf.convert_to_tensor(history_titles, dtype=tf.int32)

    logging.info(f"Converted history_titles to tensor")
    return history_tensor, history_article_ids, original_history_len

# Global cache for candidate pool for a given cutoff_time.
CANDIDATE_POOL_CACHE = {}

def precompute_candidate_pool(behaviors_df, cutoff_time):
    """
    Precompute the candidate pool for the given cutoff_time.
    This function processes the entire behaviors_df only once.
    """
    # Convert times once
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff_time)
    if cutoff_dt.tzinfo is not None:
        cutoff_dt = cutoff_dt.tz_convert(None)
    
    first_interactions = {}
    for _, row in behaviors_df.iterrows():
        if pd.isna(row["Time"]) or pd.isna(row["Impressions"]) or row["Impressions"].strip() == "":
            continue
        if row["Time"] <= np.datetime64(cutoff_dt):
            for imp in row["Impressions"].split():
                art_id, _ = imp.split('-')
                # Record the earliest time seen for the article.
                if art_id not in first_interactions or row["Time"] < first_interactions[art_id]:
                    first_interactions[art_id] = row["Time"]
    candidate_pool = [art for art, t in first_interactions.items() if t <= cutoff_dt]
    return candidate_pool

def get_candidate_pool_for_user(user_id, behaviors_df, news_df, cutoff_time, user_history_ids):
    """
    Retrieve or compute the candidate pool for the given cutoff_time, and then remove user-specific history.
    """
    global CANDIDATE_POOL_CACHE
    if cutoff_time not in CANDIDATE_POOL_CACHE:
        CANDIDATE_POOL_CACHE[cutoff_time] = precompute_candidate_pool(behaviors_df, cutoff_time)
    candidate_pool = CANDIDATE_POOL_CACHE[cutoff_time]
    candidate_pool_user = list(set(candidate_pool) - set(user_history_ids))
    return candidate_pool_user

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
    logging.info(f"Filtering based on cutoff time: {cutoff_time}")
    if cutoff_time is not None:
        candidate_pool = get_candidate_pool_for_user(user_id, behaviors_df, news_df, cutoff_time, user_history_ids)
    else:
        candidate_pool = news_df['NewsID'].unique().tolist()

    logging.info(f"After Filtering candidate pool length: {len(candidate_pool)}")
    # remove user's history from the candidate pool
    candidate_pool = list(set(candidate_pool) - set(user_history_ids))
    logging.info(f"After removing user history from candidate pool: {len(candidate_pool)}")
    # Build candidates_df
    candidates_df = news_df[news_df['NewsID'].isin(candidate_pool)].copy()
    logging.info(f"built candidates_df")

    logging.info(f"tfidf filtering candidates when min_tfidf_similarity={min_tfidf_similarity}")
    if tfidf_vectorizer is not None and min_tfidf_similarity > 0.0:
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
    logging.info(f"tfidf filtering done.")

    logging.info(f"filtering candidates (length:{len(candidates_df)}) with max size:{max_candidates}")
    if max_candidates > 0 and len(candidates_df) > max_candidates:
        candidates_df = candidates_df.head(max_candidates)

    logging.info(f"building candidate tensors")
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

def generate_base_predictions(history_tensor, candidate_tensors, models_dict, batch_size=128):
    """
    For a given user, generate predictions from each base model.
    Returns:
      - A dictionary mapping model keys to their predictions (as numpy arrays).
    """
    separate_scores = {}
    for key, model in models_dict.items():
        preds = score_candidates_in_batch(history_tensor, candidate_tensors, model, batch_size)
        separate_scores[key] = preds  # shape (num_candidates,)
    return separate_scores

def save_incremental(prediction, filename='scores.pkl'):
    with open(filename, 'ab') as f:  # Open in append mode
        pickle.dump(prediction, f)
from joblib import Parallel, delayed

def process_user_for_meta(user_id, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time):
    # Build user profile and candidate pool.
    user_history_tensor, user_history_ids, _ = build_user_profile_tensor(user_id, behaviors_df, news_df, cutoff_time, tokenizer)
    candidate_tensors, candidate_ids = user_candidate_generation(user_id, user_history_ids, behaviors_df, news_df, tokenizer, tfidf_vectorizer, cutoff_time, 0.00)
    
    # Expand dims to match model input.
    user_history_tensor = tf.expand_dims(user_history_tensor, axis=0)
    
    # Generate base model predictions.
    base_preds = generate_base_predictions(user_history_tensor, candidate_tensors, models_dict)
    
    # Combine predictions into a feature matrix.
    # Assuming candidate_ids are in the same order for all models.
    features = np.column_stack([base_preds[key] for key in models_dict.keys()])
    
    # Get ground truth: whether each candidate was clicked.
    ground_truth = get_user_future_clicks(user_id, behaviors_df, cutoff_time)
    labels = np.array([1 if art in ground_truth else 0 for art in candidate_ids])
    
    return features, labels

def build_meta_training_data_parallel(user_ids, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time, n_jobs=-1):
    """
    Build meta-training data in parallel.
    
    Parameters:
      - user_ids: list of user IDs.
      - n_jobs: number of parallel jobs (-1 uses all cores).
      
    Returns:
      - X_meta: Combined feature matrix.
      - y_meta: Combined labels.
    """
    results = Parallel(n_jobs=n_jobs)(delayed(process_user_for_meta)(
        user_id, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time
    ) for user_id in user_ids)
    
    # Unzip results into features and labels lists.
    X_meta_list, y_meta_list = zip(*results)
    X_meta = np.vstack(X_meta_list)
    y_meta = np.hstack(y_meta_list)
    return X_meta, y_meta

def batch_predict_users2(user_ids, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time, batch_size=128):
    # Lists to hold batched candidate tensors and corresponding repeated user_history tensors.
    all_candidate_tensors = []
    all_history_tensors = []
    all_candidate_ids = []  # To keep track of candidate ids per prediction.
    
    # Loop over a batch of users (you can modify this to process the whole user_ids list in chunks)
    total_users = len(user_ids)
    for i, user_id in enumerate(user_ids):
        print(f"{i}/{total_users}:start")
        # Compute user profile and candidate pool
        user_history_tensor, user_history_ids, _ = build_user_profile_tensor(user_id, behaviors_df, news_df, cutoff_time, tokenizer)
        #print(f"{i}/{total_users}: build_user_profile_tensor done")
        candidate_tensors, candidate_ids = user_candidate_generation(user_id, user_history_ids, behaviors_df, news_df, tokenizer, tfidf_vectorizer, cutoff_time, 0.00)
        #print(f"{i}/{total_users}: user_candidate_generation done")
        if not candidate_tensors:
            continue
        num_candidates = len(candidate_tensors)
        # Expand dims of user history to shape (1, max_history_length, max_title_length)
        user_history_tensor = tf.expand_dims(user_history_tensor, axis=0)
        # Repeat the user history tensor for each candidate so that its shape becomes
        # (num_candidates, max_history_length, max_title_length)
        repeated_history = tf.repeat(user_history_tensor, repeats=num_candidates, axis=0)
        # Concatenate candidate tensors along the batch axis; each tensor is shape (1, max_title_length)
        batch_candidates = tf.concat(candidate_tensors, axis=0)
        
        all_history_tensors.append(repeated_history)
        all_candidate_tensors.append(batch_candidates)
        all_candidate_ids.extend(candidate_ids)
    
    # Now, concatenate all user history tensors and candidate tensors across users.
    if not all_history_tensors or not all_candidate_tensors:
        return None, None  # No predictions if lists are empty
    
    batched_history = tf.concat(all_history_tensors, axis=0)
    batched_candidates = tf.concat(all_candidate_tensors, axis=0)
    
    # Example: Use a single model from the models_dict (or average predictions from several)
    # Here we use the first model in the dictionary.
    model_key = list(models_dict.keys())[0]
    model = models_dict[model_key]
    
    # Predict in one batch:
    predictions = model.predict(
        {
            "history_input": batched_history,
            "candidate_input": batched_candidates
        },
        batch_size=batch_size
    )
    predictions = predictions.ravel()  # Flatten to shape (total_num_candidates,)
    return predictions, all_candidate_ids

def batch_predict_users(user_ids, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time, batch_size=128):
    """
    Process a list of user_ids in batch. For each user, build candidate tensors and run the base model predictions.
    Returns:
      - predictions: a list of dictionaries (one per user), where each dict contains predictions
                     for each base model (e.g. {"model1": array([...]), "model2": array([...])}).
      - candidate_ids_list: a list of lists; each inner list contains candidate IDs for the corresponding user.
      - user_id_order: a list of user IDs (order corresponding to predictions).
    """
    all_predictions = []
    all_candidate_ids = []
    user_id_order = []
    
    total_users = len(user_ids)
    for i, user_id in enumerate(user_ids):
        print(f"{i+1}/{total_users}: Processing user {user_id}")
        logging.info(f"{i+1}/{total_users}: Processing user {user_id}")

        # Build user history and candidate pool for the user.
        logging.info(f"Starting build_user_profile_tensor")
        user_history_tensor, user_history_ids, _ = build_user_profile_tensor(user_id, behaviors_df, news_df, cutoff_time, tokenizer)
        logging.info(f"Starting user_candidate_generation")
        candidate_tensors, candidate_ids = user_candidate_generation(user_id, user_history_ids, behaviors_df, news_df,
                                                                      tokenizer, tfidf_vectorizer, cutoff_time, 0.00)
        logging.info(f"Ended user_candidate_generation")
        if not candidate_tensors:
            continue
        # Expand dims to match model input.
        user_history_tensor = tf.expand_dims(user_history_tensor, axis=0)
        num_candidates = len(candidate_tensors)
        
        # Repeat user history tensor for each candidate.
        repeated_history = tf.repeat(user_history_tensor, repeats=num_candidates, axis=0)
        # Concatenate candidate tensors along batch axis.
        batch_candidates = tf.concat(candidate_tensors, axis=0)
        
        # For demonstration, use the first model from the models_dict (or you can aggregate across models).
        user_preds = {}
        for key, model in models_dict.items():
            preds = model.predict(
                {"history_input": repeated_history, "candidate_input": batch_candidates},
                batch_size=batch_size
            )
            # Flatten predictions to a 1D array.
            user_preds[key] = preds.ravel()
        
        all_predictions.append(user_preds)
        all_candidate_ids.append(candidate_ids)
        user_id_order.append(user_id)
        print(f"{i+1}/{total_users}: Done.")
        
    return all_predictions, all_candidate_ids, user_id_order

def build_meta_training_data(user_ids, behaviors_df, news_df, models_dict, tokenizer, 
                             tfidf_vectorizer, cutoff_time):
    """
    Build the meta-training data for a batch of user IDs.
    
    This version calls the batch prediction function once, so that we do not have
    an inner loop over users for the candidate prediction step.
    
    Returns:
      - X_meta: A 2D feature matrix where each row is the feature vector for one candidate.
      - y_meta: A 1D label vector (1 if candidate was clicked, else 0).
    """
    # Get batch predictions once.
    
    logging.info(f"Starting batch_predict_users")
    predictions_list, candidate_ids_list, user_id_order = batch_predict_users(
        user_ids, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time, batch_size=128
    )
    logging.info(f"end batch_predict_users")
    
    X_meta_list = []
    y_meta_list = []
    
    total_users = len(user_id_order)
    for idx, user_id in enumerate(user_id_order):
        candidate_ids = candidate_ids_list[idx]
        logging.info(f"{idx}/{total_users} in user_id_order")
        # predictions_list[idx] is a dictionary, e.g., {"model1": array([...]), "model2": array([...])}
        user_preds = predictions_list[idx]
        # Combine the predictions from each model into a feature vector per candidate.
        features = np.column_stack([user_preds[key] for key in models_dict.keys()])
        
        # Get ground truth for this user.
        ground_truth = get_user_future_clicks(user_id, behaviors_df, cutoff_time)
        labels = np.array([1 if art in ground_truth else 0 for art in candidate_ids])
        
        X_meta_list.append(features)
        y_meta_list.append(labels)
        print(f"Processed meta data for user {idx+1}/{total_users}")
    
    if X_meta_list:
        X_meta = np.vstack(X_meta_list)
        y_meta = np.hstack(y_meta_list)
    else:
        X_meta = np.empty((0, len(models_dict)))
        y_meta = np.empty(0)
    return X_meta, y_meta

import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

def train_meta_model_incrementally(user_ids, behaviors_df, news_df, models_dict, tokenizer, 
                                   tfidf_vectorizer, cutoff_time, batch_size=100, checkpoint_dir="checkpoints", date_str="latest"):
    """
    Incrementally trains a meta-model using an SGDClassifier with logistic loss via partial_fit.
    Saves checkpoints after processing each batch and saves the final model as well.
    
    Parameters:
      - user_ids: List of user IDs.
      - behaviors_df: DataFrame of user behaviors.
      - news_df: DataFrame of news/articles.
      - models_dict: Dictionary of pre-trained base models.
      - tokenizer: Pre-fitted Keras Tokenizer.
      - tfidf_vectorizer: Pre-fitted TfidfVectorizer.
      - cutoff_time: The cutoff time (string or datetime) for splitting historical and future interactions.
      - batch_size: Number of users per batch.
      - checkpoint_dir: Directory where checkpoint models will be saved.
      - date_str: A string (such as the current date) to be appended to the final saved filename.
      
    Returns:
      - meta_model: The final meta-model trained incrementally.
    """
    import os
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Initialize an incremental learner; use logistic loss for binary classification.
    meta_model = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=42)
    classes = np.array([0, 1])  # Define classes for the first call to partial_fit
    
    num_batches = (len(user_ids) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        batch_user_ids = user_ids[i * batch_size:(i + 1) * batch_size]
        print(f"Processing batch {i+1}/{num_batches} with {len(batch_user_ids)} users.")
        
        # Build meta-training data for this batch.
        X_batch, y_batch = build_meta_training_data(batch_user_ids, behaviors_df, news_df, 
                                                    models_dict, tokenizer, tfidf_vectorizer, cutoff_time)
        if X_batch.shape[0] == 0:
            print(f"Batch {i+1} produced no candidate data, skipping.")
            continue
        
        # Update the meta-model using partial_fit.
        if i == 0:
            meta_model.partial_fit(X_batch, y_batch, classes=classes)
        else:
            meta_model.partial_fit(X_batch, y_batch)
        
        # Optionally, evaluate the current model on the batch.
        batch_auc = roc_auc_score(y_batch, meta_model.predict_proba(X_batch)[:, 1])
        print(f"After batch {i+1}: Batch AUC = {batch_auc:.4f}")
        
        # Save a checkpoint for the current model.
        checkpoint_filename = os.path.join(checkpoint_dir, f"meta_model_checkpoint_batch_{i+1}.pkl")
        with open(checkpoint_filename, "wb") as f:
            pickle.dump(meta_model, f)
        print(f"Saved checkpoint for batch {i+1} to {checkpoint_filename}")
    
    # Save the final meta-model.
    final_model_filename = f"latest_meta_model_{date_str}.pkl"
    with open(final_model_filename, "wb") as f:
        pickle.dump(meta_model, f)
    print(f"Saved final meta-model to {final_model_filename}")
    
    return meta_model


def train_meta_models_for_batches(user_ids, behaviors_df, news_df, models_dict, tokenizer, 
                                  tfidf_vectorizer, cutoff_time, batch_size=100):
    """
    Trains a separate meta model on each batch of users.
    
    For each batch:
      - Builds meta-training data using build_meta_training_data().
      - Trains a LogisticRegression meta-model on the batch.
      - Saves the trained meta-model as 'meta_model_batch_<i>.pkl'.
    
    Finally, saves the meta-model from the latest batch as 'latest_meta_model.pkl'.

    Parameters:
      - user_ids: List of user IDs to process.
      - behaviors_df: DataFrame of user behaviors.
      - news_df: DataFrame of news/articles.
      - models_dict: Dictionary of pre-trained base models.
      - tokenizer: Pre-fitted Keras Tokenizer.
      - tfidf_vectorizer: Pre-fitted TfidfVectorizer.
      - cutoff_time: Cutoff time (string or datetime) used for splitting historical and future data.
      - batch_size: The number of users to process per batch.
    
    Returns:
      - meta_models: A list containing the meta-models trained on each batch.
    """
    meta_models = []
    num_batches = (len(user_ids) + batch_size - 1) // batch_size
    X_total = []
    y_total = []
    for i in range(num_batches):
        batch_user_ids = user_ids[i * batch_size:(i + 1) * batch_size]
        print(f"Processing batch {i+1}/{num_batches} with {len(batch_user_ids)} users.")
        logging.info(f"Processing batch {i+1}/{num_batches} with {len(batch_user_ids)} users.")
        
        # Build meta-training data for this batch
        X_batch, y_batch = build_meta_training_data(batch_user_ids, behaviors_df, news_df, 
                                                    models_dict, tokenizer, tfidf_vectorizer, cutoff_time)
        if X_batch.shape[0] == 0:
            print(f"Batch {i+1} has no candidate data, skipping.")
            continue
        
        # Train meta-model for this batch
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(X_batch, y_batch)
        batch_auc = roc_auc_score(y_batch, meta_model.predict_proba(X_batch)[:, 1])
        print(f"Trained meta-model for batch {i+1}/{num_batches} - Batch AUC: {batch_auc:.4f}")
        
        meta_models.append(meta_model)
        
        # Save the meta-model for the current batch.
        batch_model_filename = f"meta_model_batch_{i+1}.pkl"
        with open(batch_model_filename, "wb") as f:
            pickle.dump(meta_model, f)
        print(f"Saved meta-model for batch {i+1} to {batch_model_filename}")
        X_total.extend(X_batch)
        y_total.extend(y_batch)
    
    if meta_models:
        # Save the latest meta model to a file for later loading.
        with open(f"latest_meta_model_{date_str}.pkl", "wb") as f:
            pickle.dump(meta_models[-1], f)
        print("Saved latest meta-model to latest_meta_model.pkl")
    else:
        print("No meta-models were trained.")

    return meta_models, 

# Example usage:
# user_ids_train = list(train_data['UserID'].unique())
# meta_models = train_meta_models_for_batches(user_ids_train, behaviors_df, news_df, models_dict,
#                                              tokenizer, tfidf_vectorizer, cutoff_time_str, batch_size=100)

def build_meta_training_data_batched(user_ids, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time, batch_size=128):
    """
    Batched version: Builds meta training data by batching model predictions for efficiency.
    """
    X_meta, y_meta = [], []
    total_users = len(user_ids)
    
    # Process users in batches
    for batch_start in range(0, total_users, batch_size):
        batch_user_ids = user_ids[batch_start:batch_start + batch_size]

        all_history_tensors = []
        all_candidate_tensors = []
        all_candidate_ids = []
        all_ground_truth = []

        for user_id in batch_user_ids:
            user_history_tensor, user_history_ids, _ = build_user_profile_tensor(
                user_id, behaviors_df, news_df, cutoff_time, tokenizer)
            candidate_tensors, candidate_ids = user_candidate_generation(
                user_id, user_history_ids, behaviors_df, news_df, tokenizer, tfidf_vectorizer, cutoff_time, 0.02)
            
            if not candidate_tensors:
                continue  # skip empty candidate set

            num_candidates = len(candidate_tensors)
            user_history_tensor = tf.expand_dims(user_history_tensor, axis=0)
            repeated_history = tf.repeat(user_history_tensor, repeats=num_candidates, axis=0)
            batch_candidates = tf.concat(candidate_tensors, axis=0)

            all_history_tensors.append(repeated_history)
            all_candidate_tensors.append(batch_candidates)
            all_candidate_ids.extend(candidate_ids)

            # Ground truth per candidate
            future_clicks = get_user_future_clicks(user_id, behaviors_df, cutoff_time)
            labels = [1 if cid in future_clicks else 0 for cid in candidate_ids]
            all_ground_truth.extend(labels)

        if not all_candidate_tensors:
            continue

        # Combine all users’ data for batch prediction
        batched_history = tf.concat(all_history_tensors, axis=0)
        batched_candidates = tf.concat(all_candidate_tensors, axis=0)

        # Run each base model and collect predictions
        base_preds = {
            model_key: models_dict[model_key].predict(
                {"history_input": batched_history, "candidate_input": batched_candidates}, batch_size=128
            ).reshape(-1)
            for model_key in models_dict
        }

        # Combine model predictions into feature matrix
        features = np.column_stack([base_preds[key] for key in models_dict.keys()])
        labels = np.array(all_ground_truth)

        X_meta.append(features)
        y_meta.append(labels)

        print(f"{min(batch_start + batch_size, total_users)}/{total_users} users done.")

    # Final output
    X_meta = np.vstack(X_meta)
    y_meta = np.hstack(y_meta)
    return X_meta, y_meta

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Example usage:
# user_ids_train = list_of_train_user_ids  # Provide a list of user IDs from your training set.
# meta_model, X_meta, y_meta = train_meta_model(user_ids_train, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time)
def train_meta_model(user_ids, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time):
    """
    Build meta-training data from the given user IDs and train a meta-model to combine base model predictions.
    
    Parameters:
      - user_ids: List of user IDs (from the training set).
      - behaviors_df: DataFrame containing user behaviors.
      - news_df: DataFrame containing news/articles.
      - models_dict: Dictionary of pre-trained base models.
      - tokenizer: Pre-fitted Keras Tokenizer.
      - tfidf_vectorizer: Pre-fitted TfidfVectorizer.
      - cutoff_time: Cutoff time (string or datetime) used for splitting historical and future data.
    
    Returns:
      - meta_model: The trained meta-model (e.g. LogisticRegression).
      - X_meta: Feature matrix used for meta-training.
      - y_meta: Ground truth binary labels.
    """
    X_meta, y_meta = build_meta_training_data(user_ids, behaviors_df, news_df, models_dict, tokenizer, tfidf_vectorizer, cutoff_time)
    
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_meta, y_meta)
    
    train_auc = roc_auc_score(y_meta, meta_model.predict_proba(X_meta)[:, 1])
    print(f"Meta model training AUC: {train_auc:.4f}")
    
    return meta_model, X_meta, y_meta
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

def incremental_train_meta_model(user_ids, behaviors_df, news_df, models_dict, 
                                 tokenizer, tfidf_vectorizer, cutoff_time, batch_size=100):
    """
    Incrementally trains the meta-model using batches of user IDs.
    
    Parameters:
      - user_ids: List of user IDs for which to build meta-training data.
      - behaviors_df: DataFrame of user behavior logs.
      - news_df: DataFrame of news/articles.
      - models_dict: Dictionary of base models.
      - tokenizer: Pre-fitted Keras Tokenizer.
      - tfidf_vectorizer: Pre-fitted TfidfVectorizer.
      - cutoff_time: The cutoff time (string or datetime) for splitting historical/future interactions.
      - batch_size: Number of users to process per batch.
    
    Returns:
      - meta_model: The trained incremental meta-model.
    """
    # Use SGDClassifier with logistic loss, which supports partial_fit
    meta_model = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
    # Define class labels required by partial_fit
    classes = np.array([0, 1])
    X_batch_total = []
    y_batch_total = []
    total_users = len(user_ids)
    for start in range(0, total_users, batch_size):
        batch_ids = user_ids[start:start + batch_size]
        # Build meta-training data for this batch:
        X_batch, y_batch = build_meta_training_data(batch_ids, behaviors_df, news_df, 
                                                    models_dict, tokenizer, tfidf_vectorizer, cutoff_time)
        if start == 0:
            # For the first batch, we need to pass the classes to partial_fit.
            meta_model.partial_fit(X_batch, y_batch, classes=classes)
        else:
            meta_model.partial_fit(X_batch, y_batch)
        
        # Optionally evaluate performance on the current batch:
        batch_auc = roc_auc_score(y_batch, meta_model.predict_proba(X_batch)[:, 1])
        print(f"Processed batch {(start // batch_size) + 1}/{(total_users // batch_size) + 1}: Batch AUC = {batch_auc:.4f}")
        logging.info(f"Processed batch {(start // batch_size) + 1}/{(total_users // batch_size) + 1}: Batch AUC = {batch_auc:.4f}")
        X_batch_total.extend(X_batch)
        y_batch_total.extend(y_batch)
    
    return meta_model, X_batch_total, y_batch_total

def score_candidates_batch(history_tensor, candidate_tensors, models_dict, batch_size=128):
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
    return separate_scores

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

def candidate_pool_from_behavior(user_id, behaviors_df, cutoff_time):
    """
    Return the set of article IDs that appear in the user's impression data after the cutoff,
    representing articles that were shown to the user.
    """
    behaviors_df['Time'] = pd.to_datetime(behaviors_df['Time'], errors='coerce')
    cutoff_dt = pd.to_datetime(cutoff_time)
    pool = set()
    user_rows = behaviors_df[behaviors_df['UserID'] == user_id]
    for _, row in user_rows.iterrows():
        if row['Time'] > np.datetime64(cutoff_dt) and pd.notna(row["Impressions"]) and row["Impressions"].strip() != "":
            for imp in row["Impressions"].split():
                try:
                    art, _ = imp.split('-')
                    pool.add(art)
                except Exception as e:
                    print(f"Error parsing impression {imp}: {e}")
    return pool

def evaluate_candidate_pool(scores, candidate_ids, ground_truth_ids, k_values):
    """
    For a given candidate set and corresponding scores, compute evaluation metrics.
    The candidate_ids and scores are assumed to be aligned.
    Returns a dictionary mapping each k to a sub-dictionary of metrics.
    """
    metrics = {}
    # Ensure that for each k we do not request more items than available.
    for k in k_values:
        effective_k = min(k, len(candidate_ids))
        if effective_k == 0:
            metrics[k] = {"recommended_ids": [],
                          "precision": 0.0,
                          "recall": 0.0,
                          "ap": 0.0,
                          "ndcg": 0.0,
                          "num_recommendations": 0}
            continue
        # Get the indices of the top effective_k scores.
        top_indices = np.argsort(scores)[-effective_k:][::-1]
        recommended_ids = [candidate_ids[i] for i in top_indices]
        prec, rec = compute_precision_recall_at_k(recommended_ids, ground_truth_ids, effective_k, candidate_ids)
        ap = average_precision_at_k(recommended_ids, ground_truth_ids, effective_k, candidate_ids)
        dcg_val = dcg_at_k(recommended_ids, ground_truth_ids, effective_k, candidate_ids)
        ideal_dcg = dcg_at_k(list(ground_truth_ids), ground_truth_ids, effective_k, candidate_ids)
        ndcg = dcg_val / ideal_dcg if ideal_dcg > 0 else 0.0
        metrics[k] = {
            "recommended_ids": recommended_ids,
            "precision": prec,
            "recall": rec,
            "ap": ap,
            "ndcg": ndcg,
            "num_recommendations": len(recommended_ids)
        }
    return metrics


def run_cluster_experiments_user_level(cluster_mapping, train_data, test_data, news_df,
                                       behaviors_df, models_dict, tokenizer,
                                       tfidf_vectorizer, cutoff_time, 
                                       k_values=[5, 10, 20, 50],
                                       partial_csv="user_level_partial_results.csv",
                                       shuffle_clusters=False, cluster_order=None):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from traceback import format_exc
    logging.info(f"partial_csv: {partial_csv}, cluster_mapping:{cluster_mapping}")
    results = []
    total_articles = len(news_df)
    
    # Initialize a partial buffer for each model including ensemble ("bagging")
    partial_buffer = {"bagging": []}
    for model_key in models_dict.keys():
        partial_buffer[model_key] = []

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
        cluster_recs = []  # For coverage (using top-10 recommendations)
        total_users = len(user_list)

        for i, user_id in enumerate(tqdm(user_list, desc=f"Evaluating users in cluster {cluster_id}")):
            try:
                logging.info(f"Starting user {user_id} in cluster {cluster_id} (index {i})")
                # 1) Build user profile.
                user_history_tensor, user_history_ids, original_history_len = build_user_profile_tensor(
                    user_id, behaviors_df, news_df, cutoff_time, tokenizer)
                num_history_articles = len(user_history_ids)
                
                # 2) Build full candidate pool.
                candidate_tensors, candidate_ids = user_candidate_generation(
                    user_id, user_history_ids, train_data, news_df,
                    tokenizer, tfidf_vectorizer, cutoff_time, 0.02)
                num_candidates = len(candidate_ids)
                
                # 3) Expand dims for prediction.
                user_history_tensor = tf.expand_dims(user_history_tensor, axis=0)
                
                # 4) Score candidates using ensemble.
                candidate_scores, separate_scores = score_candidates_ensemble_batch(
                    user_history_tensor, candidate_tensors, models_dict, batch_size=512)
                # Get ground truth clicks and shown articles.
                user_future_clicks = get_user_future_clicks(user_id, test_data, cutoff_time)
                shown_articles = get_user_shown_articles(user_id, test_data, cutoff_time)
                # Build behavior-only candidate pool from impressions.
                behavior_candidate_ids = list(candidate_pool_from_behavior(user_id, test_data, cutoff_time))
                
                # Prepare base result row.
                partial_result_row = {
                    "cluster_id": cluster_id,
                    "user_id": user_id,
                    "user_index_in_cluster": i,
                    "num_candidates_full": num_candidates,
                    "num_history_articles": num_history_articles,
                    "original_history_len": original_history_len,
                    "num_future_clicks": len(user_future_clicks),
                    "num_ground_truth_all": len(user_future_clicks),
                    "num_shown_articles": len(shown_articles),
                    "num_candidates_behavior": len(behavior_candidate_ids),
                    "num_ground_truth_in_behavior": len(set(user_future_clicks).intersection(behavior_candidate_ids))
                }
                
                # Compute metrics for the full candidate pool.
                full_metrics = evaluate_candidate_pool(candidate_scores, candidate_ids, user_future_clicks, k_values)
                
                # Compute metrics for the behavior-only candidate pool:
                # Filter scores and candidate_ids for those in the behavior-only pool.
                behavior_indices = [idx for idx, art in enumerate(candidate_ids) if art in behavior_candidate_ids]
                if behavior_indices:
                    behavior_scores = [candidate_scores[idx] for idx in behavior_indices]
                    behavior_candidate_ids_filtered = [candidate_ids[idx] for idx in behavior_indices]
                    behavior_metrics = evaluate_candidate_pool(behavior_scores, behavior_candidate_ids_filtered, user_future_clicks, k_values)
                else:
                    # If no behavior candidates, return zeros.
                    behavior_metrics = {k: {"precision": 0.0, "recall": 0.0, "ap": 0.0, "ndcg": 0.0, "num_recommendations": 0} for k in k_values}
                
                # Merge full and behavior-only metrics into the result row.
                for k in k_values:
                    partial_result_row[f"precision_full_{k}"] = full_metrics[k]["precision"]
                    partial_result_row[f"recall_full_{k}"] = full_metrics[k]["recall"]
                    partial_result_row[f"ap_full_{k}"] = full_metrics[k]["ap"]
                    partial_result_row[f"ndcg_full_{k}"] = full_metrics[k]["ndcg"]
                    partial_result_row[f"num_recommendations_full_{k}"] = full_metrics[k]["num_recommendations"]
                    
                    partial_result_row[f"precision_behavior_{k}"] = behavior_metrics[k]["precision"]
                    partial_result_row[f"recall_behavior_{k}"] = behavior_metrics[k]["recall"]
                    partial_result_row[f"ap_behavior_{k}"] = behavior_metrics[k]["ap"]
                    partial_result_row[f"ndcg_behavior_{k}"] = behavior_metrics[k]["ndcg"]
                    partial_result_row[f"num_recommendations_behavior_{k}"] = behavior_metrics[k]["num_recommendations"]
                
                partial_result_row["status"] = "DONE"
                partial_buffer["bagging"].append(partial_result_row)
                # Flush partial results every 10 users.
                if (i + 1) % 1 == 0:
                    logging.info(f"Writing partial rows to {partial_csv}")
                    save_incremental(separate_scores, f"{partial_csv.split('.csv')[0]}.pkl")
                    write_partial_rows(partial_buffer["bagging"], partial_csv)
                    partial_buffer["bagging"] = []
                
                # If desired, repeat similar metric evaluation for individual models.
                for model_key, single_model in models_dict.items():
                    ind_metrics = evaluate_candidate_pool(separate_scores[model_key], candidate_ids, user_future_clicks, k_values)
                    behavior_ind = None
                    behavior_indices = [idx for idx, art in enumerate(candidate_ids) if art in behavior_candidate_ids]
                    if behavior_indices:
                        behavior_scores = [separate_scores[model_key][idx] for idx in behavior_indices]
                        behavior_candidate_ids_filtered = [candidate_ids[idx] for idx in behavior_indices]
                        behavior_ind = evaluate_candidate_pool(behavior_scores, behavior_candidate_ids_filtered, user_future_clicks, k_values)
                    else:
                        behavior_ind = {k: {"precision": 0.0, "recall": 0.0, "ap": 0.0, "ndcg": 0.0, "num_recommendations": 0} for k in k_values}
                    
                    ind_result = {
                        "cluster_id": cluster_id,
                        "user_id": user_id,
                        "user_index_in_cluster": i,
                        "num_candidates_full": num_candidates,
                        "num_history_articles": num_history_articles,
                        "original_history_len": original_history_len,
                        "num_future_clicks": len(user_future_clicks),
                        "num_ground_truth_all": len(user_future_clicks),
                        "num_shown_articles": len(shown_articles),
                        "num_candidates_behavior": len(behavior_candidate_ids),
                        "num_ground_truth_in_behavior": len(set(user_future_clicks).intersection(behavior_candidate_ids))
                    }
                    for k in k_values:
                        ind_result[f"precision_full_{k}"] = ind_metrics[k]["precision"]
                        ind_result[f"recall_full_{k}"] = ind_metrics[k]["recall"]
                        ind_result[f"ap_full_{k}"] = ind_metrics[k]["ap"]
                        ind_result[f"ndcg_full_{k}"] = ind_metrics[k]["ndcg"]
                        ind_result[f"num_recommendations_full_{k}"] = ind_metrics[k]["num_recommendations"]
                        
                        ind_result[f"precision_behavior_{k}"] = behavior_ind[k]["precision"]
                        ind_result[f"recall_behavior_{k}"] = behavior_ind[k]["recall"]
                        ind_result[f"ap_behavior_{k}"] = behavior_ind[k]["ap"]
                        ind_result[f"ndcg_behavior_{k}"] = behavior_ind[k]["ndcg"]
                        ind_result[f"num_recommendations_behavior_{k}"] = behavior_ind[k]["num_recommendations"]
                    ind_result["status"] = "DONE"
                    partial_buffer[model_key].append(ind_result)
                    if (i + 1) % 1 == 0:
                        logging.info(f"Writing partial rows to {model_key}_{partial_csv}")

                        save_incremental(separate_scores, f"{model_key}_{partial_csv.split('.csv')[0]}.pkl")
                        write_partial_rows(partial_buffer[model_key], f"{model_key}_{partial_csv}")

                        partial_buffer[model_key] = []
                    
            except Exception as e:
                logging.error(f"Failed on user {user_id} in cluster {cluster_id} with error: {e}\n{format_exc()}")
                partial_buffer["bagging"].append({
                    "cluster_id": cluster_id,
                    "user_id": user_id,
                    "user_index_in_cluster": i,
                    "status": f"FAILED: {e}"
                })
                continue

        for key in partial_buffer:
            if partial_buffer[key]:
                write_partial_rows(partial_buffer[key], partial_csv)
                partial_buffer[key] = []
        
        # (Aggregation of cluster-level metrics omitted for brevity)
        # You would compute average precision/recall, MAP, nDCG per cluster and add to results.
    
    # Finally, save and return your results DataFrame.
    results_df = pd.DataFrame(results)
    results_df.to_csv("user_level_experiment_results.csv", index=False)
    print("Cluster-level experiment results saved to 'user_level_experiment_results.csv'")
    return results_df

# --- [Data Preparation Function] ---
def prepare_train_df(
    data_dir,
    news_file,
    behaviors_file,
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
    behaviors_path = os.path.join(data_dir, behaviors_file)
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
            behaviors_file=behaviors_file,
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
            behaviors_file=behaviors_file,
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

def run_meta_training(dataset='train', process_dfs=False, process_behaviors=False,
         data_dir_train='dataset/train/', data_dir_valid='dataset/valid/',
         zip_file_train="MINDlarge_train.zip", zip_file_valid="MINDlarge_dev.zip",
         user_category_profiles_path='', user_cluster_df_path='', cluster_id=None,):
    """
    Standalone function to train the meta-model using only the meta-training pipeline.
    
    Steps:
      1. Load the dataset and prepare the tokenizer.
      2. Build a TF-IDF vectorizer on the news combined text.
      3. Split the behaviors data to extract training users.
      4. Load or train the base models.
      5. Build meta-training data (features and labels) using the base models’ predictions.
      6. Train and return a meta-model.
      
    Returns:
      - meta_model: The trained meta-model.
      - X_meta: Feature matrix used for meta-training.
      - y_meta: Ground truth binary labels.
    """
    if dataset.lower() == 'train':
        data_dir = data_dir_train
    elif dataset.lower() == 'valid':
        data_dir = data_dir_valid
    else:
        raise ValueError("dataset must be either 'train' or 'valid'")
    # Load dataset and prepare tokenizer
    news_df, behaviors_df = init_dataset(data_dir_train)
    tokenizer, vocab_size = prepare_tokenizer(news_df)
    max_history_length = 50
    max_title_length = 30

    # Compute cutoff time (using the midpoint from behaviors)
    midpoint_time = get_midpoint_time(behaviors_df)
    cutoff_time_str = midpoint_time.isoformat().replace('+00:00', 'Z')

    # Build TF-IDF vectorizer on news CombinedText
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(news_df["CombinedText"])

    # Split behaviors to get training data (ignore test split since we focus on meta-training)
    train_data, _ = train_test_split_time(behaviors_df, cutoff_time_str)
    
    # Get unique user IDs from training data
    user_ids_train = list(train_data['UserID'].unique())
    
    # Load or train your base models
    models_dict, news_df, behaviors_df, tokenizer = get_models(
        process_dfs, process_behaviors, data_dir_train, data_dir_valid, zip_file_train, zip_file_valid
    )
    logging.info(f"computing or loading global average user profile: get_or_compute_global_average_profile")
    global_average_profile = get_or_compute_global_average_profile(behaviors_df, news_df, tokenizer, cutoff_time_str)
    GLOBAL_AVG_PROFILE = global_average_profile
    logging.info(f"Starting train_meta_model_incrementally")
    meta_model = train_meta_model_incrementally(user_ids_train, behaviors_df, news_df, models_dict, tokenizer, 
                                  tfidf_vectorizer, cutoff_time_str)
    #meta_model, X_meta, y_meta = incremental_train_meta_model(user_ids_train, behaviors_df, news_df, models_dict, 
    #                             tokenizer, tfidf_vectorizer, cutoff_time_str)
    # Build meta-training data using your existing function.
    # This returns X_meta (features: base model predictions) and y_meta (binary click labels)
    #X_meta, y_meta = build_meta_training_data(user_ids_train, behaviors_df, news_df,
    #                                          models_dict, tokenizer, tfidf_vectorizer,
    #                                          cutoff_time_str)
    
    # Train a meta-model (e.g. Logistic Regression)
    #meta_model = LogisticRegression(max_iter=1000)
    #meta_model.fit(X_meta, y_meta)
    
    # Optionally, evaluate meta-model performance on the training set
    #train_auc = roc_auc_score(y_meta, meta_model.predict_proba(X_meta)[:, 1])
    #print(f"Meta model training AUC: {train_auc:.4f}")
    
    return meta_model

# ===================== __main__ =====================
def main(dataset='train', process_dfs=False, process_behaviors=False,
         data_dir_train='dataset/train/', data_dir_valid='dataset/valid/',
         zip_file_train="MINDlarge_train.zip", zip_file_valid="MINDlarge_dev.zip",
         user_category_profiles_path='', user_cluster_df_path='', cluster_id=None, meta_train=False):
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
